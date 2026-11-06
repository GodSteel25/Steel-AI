using System;
using System.IO;
using System.Text.Json;
using Steel_AI.NetworkDefinition;

namespace Steel_AI.Helpers
{
	internal class NetworkState
	{
		private static readonly JsonSerializerOptions Options = new JsonSerializerOptions
		{
			WriteIndented = true,
			IncludeFields = true
		};

		/// <summary> 
		/// Saves the trained NeuralNetwork instance to a file on disk. 
		/// </summary>
		public static void SaveNetworkState(NeuralNetwork network, string filePath)
		{
			try
			{
				string jsonString = JsonSerializer.Serialize(network, Options);
				File.WriteAllText(filePath, jsonString);
				Console.WriteLine($"[NetworkState] Model successfully saved to: {Path.GetFullPath(filePath)}");
			}
			catch (Exception ex)
			{
				Console.WriteLine($"[NetworkState] Error saving network state: {ex.Message}");
			}
		}

		/// <summary> 
		/// Loads a saved NeuralNetwork instance from a file on disk. 
		/// </summary> 
		public static NeuralNetwork? LoadNetworkState(string filePath)
		{
			try
			{
				if (!File.Exists(filePath))
				{
					Console.WriteLine($"[NetworkState] File not found: {filePath}");
					return null;
				}

				string jsonString = File.ReadAllText(filePath);
				NeuralNetwork? network = JsonSerializer.Deserialize<NeuralNetwork>(jsonString, Options);
				Console.WriteLine($"[NetworkState] Model successfully loaded from: {Path.GetFullPath(filePath)}");
				return network;
			}
			catch (Exception ex)
			{
				Console.WriteLine($"[NetworkState] Error loading network state: {ex.Message}");
				return null;
			}
		}
	}
}