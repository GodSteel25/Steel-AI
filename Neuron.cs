using static System.Math;

namespace Steel_AI.NetworkDefinition
{
    [Serializable]
    internal class Neuron
    {
        private readonly Random _random = new();
        public double Output { get; set; }
        public double Bias { get; set; }
        public double[] Weights { get; set; } = Array.Empty<double>();
        public double Gradient { get; set; }

        ///<summary>
        /// Parameterless constructor required for network state serialization.
        /// </summary>
        public Neuron()
        {
        }
        ///<summary>
        /// Initializes a neuron and sets up randomized weights and bias.
        /// </summary>
        /// <param name="previousLayerNeuronsNumber">Number of neurons in the preceding layer.</param>
        public Neuron(int previousLayerNeuronsNumber)
        {
            if (previousLayerNeuronsNumber > 0)
            {
                Weights = new double[previousLayerNeuronsNumber];

                for (int i = 0; i < Weights.Length; i++)
                {
                    Weights[i] = InitializeRandom() / Sqrt(previousLayerNeuronsNumber);
                }

                Bias = InitializeRandom() / Sqrt(previousLayerNeuronsNumber);
            }
        }
        /// <summary>
        /// Generates Gaussian random numbers using the Box-Muller transform.
        /// </summary>
        private double InitializeRandom()
        {
            double x1 = 1.0 - _random.NextDouble();
            double x2 = 1.0 - _random.NextDouble();
            return Sqrt(-2.0 * Log(x1)) * Cos(2.0 * PI * x2);
        }
    }
}
