using Steel_AI.NetworkAlgorithm;
using Steel_AI.NetworkDefinition;
using Steel_AI.Helpers;
using System.Diagnostics;
using System.IO.Compression;

namespace Steel_AI
{
	/// <summary>
	/// Step one: teach Steel AI to read handwritten digits (MNIST, 28x28 pixels).
	/// Uses Cooper's Algorithm, NeuralNetwork, Layer and Neuron classes unchanged.
	/// Run with: dotnet run -c Release -- digits [epochs] [hidden] [learnRate]
	/// </summary>
	internal static class DigitsProgram
	{
		public static void Run(string[] args)
		{
			int epochs = args.Length > 1 ? int.Parse(args[1]) : 3;
			int hidden = args.Length > 2 ? int.Parse(args[2]) : 100;
			double learnRate = args.Length > 3 ? double.Parse(args[3], System.Globalization.CultureInfo.InvariantCulture) : 0.3;

			Console.WriteLine("===========================================");
			Console.WriteLine(" STEEL AI: Handwritten Digits (MNIST) ");
			Console.WriteLine("===========================================\n");

			string dataDir = "data";
			var (trainX, trainY) = LoadMnist(Path.Combine(dataDir, "train-images-idx3-ubyte"), Path.Combine(dataDir, "train-labels-idx1-ubyte"));
			var (testX, testY) = LoadMnist(Path.Combine(dataDir, "t10k-images-idx3-ubyte"), Path.Combine(dataDir, "t10k-labels-idx1-ubyte"));
			Console.WriteLine($"Training images: {trainX.Length}   Held-out test images (never trained on): {testX.Length}");
			Console.WriteLine($"Network: 784 inputs -> {hidden} hidden -> 10 outputs, learn rate {learnRate}, {epochs} epochs\n");

			var network = new NeuralNetwork(784, hidden, 10);
			var algorithm = new Algorithm();
			var rng = new Random(25);
			int[] order = Enumerable.Range(0, trainX.Length).ToArray();
			var total = Stopwatch.StartNew();

			Console.WriteLine($"Before training, test accuracy: {Accuracy(network, algorithm, testX, testY):P2} (random guessing is about 10%)\n");

			for (int epoch = 1; epoch <= epochs; epoch++)
			{
				var sw = Stopwatch.StartNew();
				// Shuffle so the network doesn't see the digits in the same order every pass
				for (int i = order.Length - 1; i > 0; i--) { int j = rng.Next(i + 1); (order[i], order[j]) = (order[j], order[i]); }

				foreach (int idx in order)
				{
					algorithm.ForwardPropagation(network.Layers, trainX[idx]);
					algorithm.CalculateGradients(network.Layers, OneHot(trainY[idx]));
					algorithm.UpdateParameters(network.Layers, learnRate);
				}

				double testAcc = Accuracy(network, algorithm, testX, testY);
				Console.WriteLine($"Epoch {epoch}/{epochs} - test accuracy {testAcc:P2} - {sw.Elapsed.TotalSeconds:F1}s");
			}

			Console.WriteLine($"\nTotal training time: {total.Elapsed.TotalSeconds:F1}s");
			PrintConfusion(network, algorithm, testX, testY);
			ShowSamples(network, algorithm, testX, testY, 3);
			NetworkState.SaveNetworkState(network, "digits_model.json");
		}

		static double[] OneHot(int label)
		{
			var t = new double[10];
			t[label] = 1.0;
			return t;
		}

		static int Predict(NeuralNetwork net, Algorithm alg, double[] x)
		{
			double[] outp = alg.ForwardPropagation(net.Layers, x);
			int best = 0;
			for (int i = 1; i < outp.Length; i++) if (outp[i] > outp[best]) best = i;
			return best;
		}

		static double Accuracy(NeuralNetwork net, Algorithm alg, double[][] xs, int[] ys)
		{
			int correct = 0;
			for (int i = 0; i < xs.Length; i++) if (Predict(net, alg, xs[i]) == ys[i]) correct++;
			return (double)correct / xs.Length;
		}

		static void PrintConfusion(NeuralNetwork net, Algorithm alg, double[][] xs, int[] ys)
		{
			var perDigit = new int[10]; var perDigitTotal = new int[10];
			for (int i = 0; i < xs.Length; i++) { perDigitTotal[ys[i]]++; if (Predict(net, alg, xs[i]) == ys[i]) perDigit[ys[i]]++; }
			Console.WriteLine("\nAccuracy by digit on the held-out test set:");
			for (int d = 0; d < 10; d++) Console.WriteLine($"  {d}: {(double)perDigit[d] / perDigitTotal[d]:P1}");
		}

		static void ShowSamples(NeuralNetwork net, Algorithm alg, double[][] xs, int[] ys, int count)
		{
			Console.WriteLine("\nA few test images as ASCII art:");
			for (int s = 0; s < count; s++)
			{
				double[] x = xs[s];
				for (int r = 0; r < 28; r += 2)
				{
					var line = new System.Text.StringBuilder();
					for (int c = 0; c < 28; c++) { double v = x[r * 28 + c]; line.Append(v > 0.6 ? '#' : v > 0.2 ? '+' : ' '); }
					Console.WriteLine("  " + line);
				}
				Console.WriteLine($"  Actual: {ys[s]}   Steel AI says: {Predict(net, alg, x)}\n");
			}
		}

		/// <summary>Reads the standard MNIST IDX files and scales pixels to 0.0-1.0.</summary>
		static (double[][] images, int[] labels) LoadMnist(string imagePath, string labelPath)
		{
			using var img = new BinaryReader(File.OpenRead(imagePath));
			using var lbl = new BinaryReader(File.OpenRead(labelPath));
			int ReadBE(BinaryReader br) { var b = br.ReadBytes(4); return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]; }

			if (ReadBE(img) != 2051 || ReadBE(lbl) != 2049) throw new InvalidDataException("Not MNIST IDX files");
			int n = ReadBE(img); int rows = ReadBE(img); int cols = ReadBE(img);
			if (ReadBE(lbl) != n) throw new InvalidDataException("Image/label count mismatch");

			var images = new double[n][]; var labels = new int[n];
			for (int i = 0; i < n; i++)
			{
				byte[] px = img.ReadBytes(rows * cols);
				images[i] = new double[rows * cols];
				for (int p = 0; p < px.Length; p++) images[i][p] = px[p] / 255.0;
				labels[i] = lbl.ReadByte();
			}
			return (images, labels);
		}
	}
}