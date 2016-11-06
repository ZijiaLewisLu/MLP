import heapq
import math
import numpy as np
# from collections import namedtuple


class Caption(object):
    """Represents a complete or partial caption."""

    def __init__(self, sentence, state, logprob, score, metadata=None):
        """Initializes the Caption.

        Args:
          sentence: List of word ids in the caption.
          state: Model state after generating the previous word.
          logprob: Log-probability of the caption.
          score: Score of the caption.
          metadata: Optional metadata associated with the partial sentence. If not
            None, a list of strings with the same length as 'sentence'.
        """
        self.sentence = sentence
        self.state = state
        self.logprob = logprob
        self.score = score
        self.metadata = metadata

    def __cmp__(self, other):
        """Compares Captions by score."""
        assert isinstance(other, Caption)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1


class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.

        The only method that can be called immediately after extract() is reset().

        Args:
          sort: Whether to return the elements in descending sorted order.

        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []


def beam_search(model, vocab, sess, data, batch_size,
                beam_size=3,
                max_caption_length=20,
                length_normalization_factor=0.0,
                no_input=False):
    """
    Args:

      model: Must have methods build_model(), encode_step() and inference_step().
      vocab: A Vocabulary object.
      sess: TensorFlow Session object.
      data:

      beam_size: Beam size to use when generating captions.
      max_caption_length: The maximum caption length before stopping the search.
      length_normalization_factor: If != 0, a number x such that captions are
        scored by logprob/length^x, rather than logprob. This changes the
        relative scores of captions depending on their lengths. For example, if
        x > 0 then longer captions will be favored.

    Returns:
      A list of Caption sorted by descending score.
    """

    initial_beam = Caption(
        sentence=[],
        state=model.pointer_zero_state(),
        logprob=0.0,
        score=0.0,
        metadata=[""])
    partial_captions = TopN(beam_size)
    partial_captions.push(initial_beam)
    complete_captions = TopN(beam_size)

    # Run beam search.
    for _ in range(max_caption_length - 1):
        partial_captions_list = partial_captions.extract()
        partial_captions.reset()
        input_feed = [] if no_input else np.array(
            [c.sentence[-1] for c in partial_captions_list])
        state_feed = np.array([c.state for c in partial_captions_list])

        # run one iteration
        prediction, new_states, metadata = model.inference_step(
            sess, input_feed, state_feed)

        for i, partial_caption in enumerate(partial_captions_list):
            word_probabilities = softmax[i]
            state = new_states[i]
            # For this partial caption, get the beam_size most probable next
            # words.
            words_and_probs = list(enumerate(word_probabilities))
            words_and_probs.sort(key=lambda x: -x[1])
            words_and_probs = words_and_probs[0:beam_size]
            # Each next word gives a new partial caption.
            for w, p in words_and_probs:
                if p < 1e-12:
                    continue  # Avoid log(0).
                sentence = partial_caption.sentence + [w]
                logprob = partial_caption.logprob + math.log(p)
                score = logprob
                if metadata:
                    metadata_list = partial_caption.metadata + [metadata[i]]
                else:
                    metadata_list = None
                if w == vocab.end_id:
                    if length_normalization_factor > 0:
                        score /= len(sentence)**length_normalization_factor
                    beam = Caption(sentence, state, logprob,
                                   score, metadata_list)
                    complete_captions.push(beam)
                else:
                    beam = Caption(sentence, state, logprob,
                                   score, metadata_list)
                    partial_captions.push(beam)
        if partial_captions.size() == 0:
            # We have run out of partial candidates; happens when beam_size =
            # 1.
            break

    # If we have no complete captions then fall back to the partial captions.
    # But never output a mixture of complete and partial captions because a
    # partial caption could have a higher score than all the complete captions.
    if not complete_captions.size():
        complete_captions = partial_captions

    return complete_captions.extract(sort=True)
