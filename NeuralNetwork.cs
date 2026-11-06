namespace Steel_AI.NetworkDefinition
{
    [Serializable]
    internal class NeuralNetwork
    {
        /// <summary> 
        /// Array of layers representing the entire neural network structure.
        /// </summary>
        public Layer[] Layers { get; set; } = Array.Empty<Layer>();

        /// <summary>
        /// Parameterless constructor required for serialization.
        /// </summary>
        public NeuralNetwork()
        {
        }

        /// <summary>
        /// Initializes a neural network with specified layer dimensions.
        /// </summary>
        /// <param name="layersData">Array of neuron counts per layer (e.g., 784, 350, 10).</param>
        public NeuralNetwork(params int[] layersData)
        {
            if (layersData == null || layersData.Length == 0) return;

            Layers = new Layer[layersData.Length];

            // First layer (Input Layer) has now incoming weight connections
            Layers[0] = new Layer(layersData[0], 0);

            // Construct hidden and output layers, connecting to the preceding layer size
            for (int i = 1; i < layersData.Length; i++)
            {
                Layers[i] = new Layer(layersData[i], layersData[i - 1]);
            }
        }
    }
}
