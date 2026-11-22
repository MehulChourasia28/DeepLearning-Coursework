"""Unit checks for preprocessing, models and metrics on tiny synthetic inputs."""
import unittest

import torch

from scripts.crisis_models import (TextCNN, Vocabulary, build_model, classification_metrics,
                                   count_trainable, tokenize)


class CrisisModelTests(unittest.TestCase):
    def test_tokenize(self):
        text = "RT @Cal_OES: Fire DESTROYS homes!! https://t.co/abc #Harvey we're ok"
        self.assertEqual(tokenize(text), ["<user>", "fire", "destroys", "homes", "!", "!", "<url>",
                                          "harvey", "we're", "ok"])

    def test_vocabulary_is_training_only_and_padded(self):
        vocab = Vocabulary.build([["a", "b"], ["a", "c"]], min_freq=2)
        self.assertEqual(vocab.itos, ["<pad>", "<unk>", "a"])
        ids, lengths, truncated = vocab.encode([["a", "z", "a"], []], max_tokens=2)
        self.assertEqual(ids.tolist(), [[2, 1], [1, 0]])
        self.assertEqual(lengths.tolist(), [2, 1])
        self.assertEqual(truncated, 1)

    def test_metrics_match_hand_calculation(self):
        # labels: 0,0,1,1,1  preds: 0,1,1,1,0 -> class0 P=1/2 R=1/2; class1 P=2/3 R=2/3
        m = classification_metrics(torch.tensor([0, 0, 1, 1, 1]), torch.tensor([0, 1, 1, 1, 0]), 2)
        self.assertEqual(m["confusion_matrix"], [[1, 1], [1, 2]])
        self.assertAlmostEqual(m["accuracy"], 0.6)
        self.assertAlmostEqual(m["macro_f1"], (0.5 + 2 / 3) / 2)
        self.assertAlmostEqual(m["precision"][1], 2 / 3)
        # A class never predicted has F1 0 and still counts in the macro average.
        m = classification_metrics(torch.tensor([0, 1, 1]), torch.tensor([1, 1, 1]), 2)
        self.assertEqual(m["f1"][0], 0.0)
        self.assertAlmostEqual(m["macro_f1"], 0.8 / 2)

    def test_textcnn_ignores_padding(self):
        torch.manual_seed(0)
        encoder = TextCNN(10, embed_dim=8, kernel_sizes=(2, 3), filters=4, dropout=0.0).eval()
        short = torch.tensor([[3, 4, 5, 0, 0, 0]])
        noisy = torch.tensor([[3, 4, 5, 7, 8, 9]])  # same text, different content after length
        length = torch.tensor([3])
        self.assertTrue(torch.equal(encoder(short, length), encoder(noisy, length)))

    def test_model_shapes_and_shared_head(self):
        batch = {"ids": torch.randint(0, 20, (4, 64)), "lengths": torch.tensor([64, 3, 1, 10]),
                 "image": torch.randn(4, 512)}
        params = {}
        for kind in ("text_textcnn", "text_bilstm", "image_mlp", "fusion_textcnn", "fusion_bilstm"):
            model = build_model(kind, vocab_size=20, num_classes=2).eval()
            self.assertEqual(model(batch).shape, (4, 2))
            params[kind] = count_trainable(model)
        # Swapping TextCNN for BiLSTM keeps the image branch and classifier head identical.
        cnn, lstm = build_model("fusion_textcnn", 20, 2), build_model("fusion_bilstm", 20, 2)
        self.assertEqual(str(cnn.head), str(lstm.head))
        self.assertEqual(str(cnn.image_branch), str(lstm.image_branch))


if __name__ == "__main__":
    unittest.main()
