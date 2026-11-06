namespace Steel_AI.NetworkDefinition
{
    [Serializable]
    internal class Layer
    {
        /// <summary>
        /// Array of neurons belonging to this layer.
        /// </summary>
        public Neuron[] Neurons { get; set; } = Array.Empty<Neuron>();

        /// <summary>
        /// Parameterless constructor required for serialization.
        /// </summary>
        public Layer()
        {
        }

        /// <summary>
        /// Initializes a network layer and populates it with neurons.
        /// </summary>
        /// <param name="neuronsNumber">Number of neurons in this layer.</param>
        /// <param name="previousLayerNeuronNumber">Number of neurons in the preceding layer.</param>
        public Layer(int neuronsNumber, int previousLayerNeuronsNumber)
        {
            Neurons = new Neuron[neuronsNumber];

            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new Neuron(previousLayerNeuronsNumber);
            }
        }
    }
}