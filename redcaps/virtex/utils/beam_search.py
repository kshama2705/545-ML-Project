r"""
This Beam Search implementation is adapted with minor modifications from
`AllenNLP <https://github.com/allenai/allennlp/blob/master/allennlp/nn/beam_search.py>`_.

Thanks to the developers of AllenNLP!
"""
from typing import Callable, List, Tuple
import warnings

import torch
from torch.nn import functional as F

class AutoRegressiveBeamSearch(object):
    r"""
    Implements the beam search algorithm for decoding the most likely captions.
    This only works for auto-regressive models (Transformer-like) and not
    recurrent models (LSTM-like).

    Parameters
    ----------
    eos_index: int
        The index of the end token (``[EOS]``) in vocabulary.
    max_steps: int, optional (default = 50)
        The maximum number of decoding steps.
    beam_size: int, optional (default = 5)
        The width of the beam used.
    per_node_beam_size: int, optional (default = 2)
        The maximum number of candidates to consider per node, at each step in
        the search. Setting this parameter to a number smaller than `beam_size`
        may give better results, as it can introduce more diversity into the
        search. See `Beam Search Strategies for Neural Machine Translation.
        Freitag and Al-Onaizan, 2017 <https://arxiv.org/abs/1702.01806>`_.
    """

    def __init__(
        self,
        eos_index: int,
        max_steps: int = 50,
        beam_size: int = 5,
        per_node_beam_size: int = 2,
    ):
        self._eos_index = eos_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or beam_size

    def search(
        self, start_predictions: torch.Tensor, step: Callable[..., torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Given a starting state and a step function, apply beam search to find
        the most likely target captions.

        Parameters
        ----------
        start_predictions : torch.Tensor
            Tensor containing the initial predictions, shape ``(batch_size, )``.
            Usually the initial predictions are just the index of the start
            token (``[SOS]``) in the vocabulary.
        step : Callable[..., torch.Tensor]
            A function that is responsible for computing the next most likely
            tokens, given the past predictions. Predictions from all previous
            timesteps are required, not just the last timestep, because our
            model is auto-regressive instead of recurrent.  The function should
            The function is expected to return a tensor of shape
            ``(group_size, target_vocab_size)`` containing
            the logits of the tokens for the next step.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of ``(predictions, logprobs)``, where ``predictions``
            has shape ``(batch_size, beam_size, max_steps)`` and ``logprobs``
            has shape ``(batch_size, beam_size)``.
        """
        batch_size = start_predictions.size()[0]

        # List of `(batch_size, beam_size)` tensors. One for each time step.
        # Does not include the start symbols, which are implicit.
        predictions: List[torch.Tensor] = []

        # List of (batch_size, beam_size) tensors. One for each time step. None
        # for the first.  Stores the index n for the parent prediction, i.e.
        # predictions[t-1][i][n], that it came from.
        backpointers: List[torch.Tensor] = []

        # Calculate the first timestep. This is done outside the main loop
        # because we are going from a single decoder input (the output from the
        # encoder) to the top `beam_size` decoder outputs. On the other hand,
        # within the main loop we are going from the `beam_size` elements of the
        # beam to `beam_size`^2 candidates from which we will select the top
        # `beam_size` elements for the next iteration.
        # shape: (batch_size, num_classes)
        start_class_logits = step(start_predictions)

        # Convert logits to logprobs.
        # shape: (batch_size * beam_size, vocab_size)
        start_class_logprobs = F.log_softmax(start_class_logits, dim=1)

        num_classes = start_class_logprobs.size()[1]

        # Make sure `per_node_beam_size` is not larger than `num_classes`.
        if self.per_node_beam_size > num_classes:
            raise ValueError(
                f"Target vocab size ({num_classes:d}) too small "
                f"relative to per_node_beam_size ({self.per_node_beam_size:d}).\n"
                f"Please decrease beam_size or per_node_beam_size."
            )

        # shape: (batch_size, beam_size), (batch_size, beam_size)
        start_top_logprobs, start_predicted_classes = start_class_logprobs.topk(
            self.beam_size
        )
        if (
            self.beam_size == 1
            and (start_predicted_classes == self._eos_index).all()
        ):
            warnings.warn(
                "Empty captions predicted. You may want to increase beam "
                "size or ensure your step function is working properly.",
                RuntimeWarning,
            )
            return start_predicted_classes.unsqueeze(-1), start_top_logprobs

        # The log probs for the last time step.
        # shape: (batch_size, beam_size)
        last_logprobs = start_top_logprobs

        # shape: [(batch_size, beam_size)]
        predictions.append(start_predicted_classes)

        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size * beam_size, num_classes)
        logprobs_after_end = start_class_logprobs.new_full(
            (batch_size * self.beam_size, num_classes), float("-inf")
        )
        logprobs_after_end[:, self._eos_index] = 0.0

        for timestep in range(self.max_steps - 1):
            # shape: (batch_size * beam_size,)
            last_predictions = predictions[-1].reshape(batch_size * self.beam_size)

            # If every predicted token from the last step is `self._eos_index`,
            # then we can stop early.
            if (last_predictions == self._eos_index).all():
                break

            # Take a step. This get the predicted log probs of the next classes.
            predictions_so_far = torch.stack(predictions).permute(1, 2, 0).view(
                batch_size * self.beam_size, -1
            )
            # shape: (batch_size * beam_size, num_classes)
            class_logits = step(predictions_so_far)

            # Convert logits to logprobs.
            # shape: (batch_size * beam_size, vocab_size)
            class_logprobs = F.log_softmax(class_logits, dim=1)

            # Set logprobs of last predicted tokens as high negative value to avoid
            # repetition in caption.
            for index in range(batch_size * self.beam_size):
                class_logprobs[index, predictions_so_far[index, -1]] = -10000

            # shape: (batch_size * beam_size, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                batch_size * self.beam_size, num_classes
            )
            # Here we are finding any beams where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the beam to predict the end token
            # this timestep as well.
            # shape: (batch_size * beam_size, num_classes)
            cleaned_logprobs = torch.where(
                last_predictions_expanded == self._eos_index,
                logprobs_after_end,
                class_logprobs,
            )
            # shape (both): (batch_size * beam_size, per_node_beam_size)
            top_logprobs, predicted_classes = cleaned_logprobs.topk(
                self.per_node_beam_size
            )
            # Here we expand the last log probs to `(batch_size * beam_size,
            # per_node_beam_size)` so that we can add them to the current log
            # probs for this timestep. This lets us maintain the log
            # probability of each element on the beam.
            # shape: (batch_size * beam_size, per_node_beam_size)
            expanded_last_logprobs = (
                last_logprobs.unsqueeze(2)
                .expand(batch_size, self.beam_size, self.per_node_beam_size)
                .reshape(batch_size * self.beam_size, self.per_node_beam_size)
            )
            # shape: (batch_size * beam_size, per_node_beam_size)
            summed_top_logprobs = top_logprobs + expanded_last_logprobs

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_summed = summed_top_logprobs.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )
            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_predicted_classes = predicted_classes.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )
            # Keep only the top `beam_size` beam indices.
            # shape: (batch_size, beam_size), (batch_size, beam_size)
            restricted_beam_logprobs, restricted_beam_indices = reshaped_summed.topk(
                self.beam_size
            )
            # Use the beam indices to extract the corresponding classes.
            # shape: (batch_size, beam_size)
            restricted_predicted_classes = reshaped_predicted_classes.gather(
                1, restricted_beam_indices
            )
            predictions.append(restricted_predicted_classes)

            # shape: (batch_size, beam_size)
            last_logprobs = restricted_beam_logprobs

            # The beam indices come from a `beam_size * per_node_beam_size`
            # dimension where the indices with a common ancestor are grouped
            # together. Hence dividing by `per_node_beam_size` gives the
            # ancestor. (Note that this is integer division as the tensor is a
            # LongTensor.)
            # shape: (batch_size, beam_size)
            backpointer = restricted_beam_indices // self.per_node_beam_size

            backpointers.append(backpointer)

        if not torch.isfinite(last_logprobs).all():
            warnings.warn(
                "Infinite log probs encountered. Some final captions may not "
                "make sense. This can happen when the beam size is larger than"
                " the number of valid (non-zero probability) transitions that "
                "the step function produces.",
                RuntimeWarning,
            )

        # Reconstruct the captions.
        # shape: [(batch_size, beam_size, 1)]
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]

        # shape: (batch_size, beam_size)
        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            # shape: (batch_size, beam_size, 1)
            cur_preds = (
                predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)
            )
            reconstructed_predictions.append(cur_preds)

            # shape: (batch_size, beam_size)
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

        # shape: (batch_size, beam_size, 1)
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)

        reconstructed_predictions.append(final_preds)

        # shape: (batch_size, beam_size, max_steps)
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)

        # Select the top-beam and its logprobs.
        all_predictions = all_predictions[:, 0, :]
        last_logprobs = last_logprobs[:, 0]

        return all_predictions, last_logprobs
