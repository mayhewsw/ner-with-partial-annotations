# pylint: disable=no-self-use,invalid-name
import itertools
import math

from pytest import approx, raises
import torch

from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase

from mylib.modules.marginal_conditional_random_field import MarginalConditionalRandomField


class TestConditionalRandomField(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.logits = torch.Tensor([
            [[0, 0, .5, .5, .2], [0, 0, .3, .3, .1], [0, 0, .9, 10, 1]],
            [[0, 0, .2, .5, .2], [0, 0, 3, .3, .1], [0, 0, .9, 1, 1]],
        ])

        self.orig_tags = torch.LongTensor([
            [2, 3, 4],
            [3, 2, 2]
        ])

        self.tags = torch.FloatTensor([
            [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
            [[0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]
        ])
        self.tags = torch.clamp(self.tags + 1e-30, max=1)

        self.transitions = torch.Tensor([
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.8, 0.3, 0.1, 0.7, 0.9],
            [-0.3, 2.1, -5.6, 3.4, 4.0],
            [0.2, 0.4, 0.6, -0.3, -0.4],
            [1.0, 1.0, 1.0, 1.0, 1.0]
        ])

        self.transitions_from_start = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.6])
        self.transitions_to_end = torch.Tensor([-0.1, -0.2, 0.3, -0.4, -0.4])

        # Use the CRF Module with fixed transitions to compute the log_likelihood
        self.crf = MarginalConditionalRandomField(5)
        self.crf.transitions = torch.nn.Parameter(self.transitions)
        self.crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        self.crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)

    def score(self, logits, tags):
        """
        Computes the likelihood score for the given sequence of tags,
        given the provided logits (and the transition weights in the CRF model)
        """
        # Start with transitions from START and to END
        total = self.transitions_from_start[tags[0]] + self.transitions_to_end[tags[-1]]
        # Add in all the intermediate transitions
        for tag, next_tag in zip(tags, tags[1:]):
            total += self.transitions[tag, next_tag]
        # Add in the logits for the observed tags
        for logit, tag in zip(logits, tags):
            total += logit[tag]
        return total

    def test_forward_works_without_mask(self):
        log_likelihood = self.crf(self.logits, self.tags).item()

        # Now compute the log-likelihood manually
        manual_log_likelihood = 0.0

        # For each instance, manually compute the numerator
        # (which is just the score for the logits and actual tags)
        # and the denominator
        # (which is the log-sum-exp of the scores for the logits across all possible tags)
        for logits_i, tags_i in zip(self.logits, self.orig_tags):
            numerator = self.score(logits_i.detach(), tags_i.detach())
            all_scores = [self.score(logits_i.detach(), tags_j)
                          for tags_j in itertools.product(range(5), repeat=3)]
            denominator = math.log(sum(math.exp(score) for score in all_scores))
            # And include them in the manual calculation.
            manual_log_likelihood += numerator - denominator

        # The manually computed log likelihood should equal the result of crf.forward.
        assert manual_log_likelihood.item() == approx(log_likelihood)

    def test_forward_works_with_mask(self):
        # Use a non-trivial mask
        mask = torch.LongTensor([
            [1, 1, 1],
            [1, 1, 0]
        ])

        log_likelihood = self.crf(self.logits, self.tags, mask).item()

        # Now compute the log-likelihood manually
        manual_log_likelihood = 0.0

        # For each instance, manually compute the numerator
        #   (which is just the score for the logits and actual tags)
        # and the denominator
        #   (which is the log-sum-exp of the scores for the logits across all possible tags)
        for logits_i, tags_i, mask_i in zip(self.logits, self.orig_tags, mask):
            # Find the sequence length for this input and only look at that much of each sequence.
            sequence_length = torch.sum(mask_i.detach())
            logits_i = logits_i.data[:sequence_length]
            tags_i = tags_i.data[:sequence_length]

            numerator = self.score(logits_i, tags_i)
            all_scores = [self.score(logits_i, tags_j)
                          for tags_j in itertools.product(range(5), repeat=sequence_length)]
            denominator = math.log(sum(math.exp(score) for score in all_scores))
            # And include them in the manual calculation.
            manual_log_likelihood += numerator - denominator

        # The manually computed log likelihood should equal the result of crf.forward.
        assert manual_log_likelihood.item() == approx(log_likelihood)

    def get_all_sequences(self, tags):
        import itertools
        newseq = []
        for marginal in tags:
            newmarginal = []
            for i, val in enumerate(marginal):
                if val > 0:
                    newmarginal.append(i)
            newseq.append(newmarginal)

        all_seqs = itertools.product(*newseq)

        return all_seqs

    def test_forward_works_with_mask_and_marginal(self):
        # Use a non-trivial mask
        mask = torch.LongTensor([
            [1, 1, 1],
            [1, 1, 0]
        ])

        self.tags = torch.FloatTensor([
            [[1, 1, 1, 1, 1], [0, 0, 0, 1, 0], [1, 1, 1, 1, 1]],
            [[0, 0, 0, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]
        ])

        self.get_all_sequences(self.tags[0].long())

        log_likelihood = self.crf(self.logits, torch.clamp(self.tags + 1e-30, max=1), mask).item()

        # Now compute the log-likelihood manually
        manual_log_likelihood = 0.0

        # For each instance, manually compute the numerator
        #   (which is just the score for the logits and actual tags)
        # and the denominator
        #   (which is the log-sum-exp of the scores for the logits across all possible tags)
        for logits_i, tags_i, mask_i in zip(self.logits, self.tags, mask):
            # Find the sequence length for this input and only look at that much of each sequence.
            sequence_length = torch.sum(mask_i.detach())
            logits_i = logits_i.data[:sequence_length]
            tags_i = tags_i.data[:sequence_length]

            all_seqs = self.get_all_sequences(tags_i)

            num_scores = []
            for seq in all_seqs:
                num_scores.append(self.score(logits_i, seq))

            numerator = math.log(sum(math.exp(score) for score in num_scores))

            all_scores = [self.score(logits_i, tags_j)
                          for tags_j in itertools.product(range(5), repeat=sequence_length)]
            denominator = math.log(sum(math.exp(score) for score in all_scores))
            # And include them in the manual calculation.
            manual_log_likelihood += numerator - denominator

        # The manually computed log likelihood should equal the result of crf.forward.
        assert manual_log_likelihood == approx(log_likelihood)