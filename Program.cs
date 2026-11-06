using Steel_AI.Helpers;
using Steel_AI.NetworkAlgorithm;
using Steel_AI.NetworkDefinition;
using System;
using System.IO;

namespace Steel_AI
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("===========================================");
            Console.WriteLine(" STEEL AI: Model Persistence Test (XOR) ");
            Console.WriteLine("===========================================\n");
            string modelPath = "xor_model.json";
            NeuralNetwork? network = null; Algorithm algorithm = new Algorithm();
            // 1\. Define XOR Dataset 
            double[][] inputs = new double[][]
            {
                new double[] { 0, 0 },
                new double[] { 0, 1 },
                new double[] { 1, 0 },
                new double[] { 1, 1 }
            };

            double[][] targets = new double[][]
            {
                new double[] { 0 },
                new double[] { 1 },
                new double[] { 1 },
                new double[] { 0 }
            };
            // 2\. Check if Saved Model Exists
            if (File.Exists(modelPath))
            {
                Console.WriteLine($"[INFO] Found saved model at '{modelPath}'. Loading weights from disk...");
                network = NetworkState.LoadNetworkState(modelPath);
            }
            // 3\. If No Saved Model Found, Train New Network and Save 
            if (network == null)
            {
                Console.WriteLine("[INFO] No saved model found. Initializing new network and training...\n");
                network = new NeuralNetwork(2, 3, 1);
                double learningRate = 0.5;
                int epochs = 20000;
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    double totalError = 0;
                    for (int i = 0; i < inputs.Length; i++)
                    {
                        double[] predictions = algorithm.ForwardPropagation(network.Layers, inputs[i]);
                        double error = targets[i][0] - predictions[0];
                        totalError += error * error;
                        algorithm.CalculateGradients(network.Layers, targets[i]);
                        algorithm.UpdateParameters(network.Layers, learningRate);
                    }
                    if ((epoch + 1) % 4000 == 0)
                    {
                        Console.WriteLine($"Epoch {epoch + 1}/{epochs} - Loss: {totalError:F6}");
                    }
                }
                // Save trained network parameters 
                Console.WriteLine("\nSaving trained model state to disk...");
                NetworkState.SaveNetworkState(network, modelPath);
            }
            // 4\. Verify Model Predictions 
            Console.WriteLine("\n-------------------------------------------");
            Console.WriteLine(" MODEL PREDICTIONS VERIFICATION ");
            Console.WriteLine("-------------------------------------------");
            for (int i = 0; i < inputs.Length; i++)
            {
                double[] output = algorithm.ForwardPropagation(network.Layers, inputs[i]);
                Console.WriteLine($"Input: [{inputs[i][0]}, {inputs[i][1]}] => Output: {output[0]:F4} | Target: {targets[i][0]}");
            }
            Console.WriteLine("\nSteel AI is a production of GodSteelGaming");
            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
        }
    }
}