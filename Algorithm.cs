using Steel_AI.NetworkDefinition;
namespace Steel_AI.NetworkAlgorithm
{
    internal class Algorithm
    {
        /// <summary> 
        /// Runs forward propagation through the network layers to generate output predictions. 
        /// </summary> 
        public double[] ForwardPropagation(Layer[] layers, double[] inputs)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                layers[0].Neurons[i].Output = inputs[i];
            }

            for (int i = 1; i < layers.Length; i++)
            {
                Layer currentLayer = layers[i];
                Layer previousLayer = layers[i - 1];

                for (int j = 0; j < currentLayer.Neurons.Length; j++)
                {
                    Neuron currentNeuron = currentLayer.Neurons[j];
                    double output = currentNeuron.Bias;

                    for (int k = 0; k < previousLayer.Neurons.Length; k++)
                    {
                        output += previousLayer.Neurons[k].Output * currentNeuron.Weights[k];
                    }

                    currentNeuron.Output = Activate(output);
                }
            }

            Layer outputLayer = layers[layers.Length - 1];
            return outputLayer.Neurons.Select(n => n.Output).ToArray();
        }

        /// <summary> 
        /// Calculates error gradients for neurons in the output layer and hidden layers. 
        /// </summary> 
        public void CalculateGradients(Layer[] layers, double[] targets)
        {
            // 1. Output Layer Gradients
            Layer outputLayer = layers[layers.Length - 1];
            for (int i = 0; i < outputLayer.Neurons.Length; i++)
            {
                Neuron neuron = outputLayer.Neurons[i];
                neuron.Gradient = (neuron.Output - targets[i]) * (neuron.Output * (1.0 - neuron.Output));
            }

            // 2. Hidden Layers Gradients (Moving backward)
            for (int layerIndex = layers.Length - 2; layerIndex >= 1; layerIndex--)
            {
                Layer currentLayer = layers[layerIndex];
                Layer nextLayer = layers[layerIndex + 1];

                for (int i = 0; i < currentLayer.Neurons.Length; i++)
                {
                    Neuron neuron = currentLayer.Neurons[i];
                    double sumGradientTimesWeight = 0.0;

                    for (int j = 0; j < nextLayer.Neurons.Length; j++)
                    {
                        Neuron nextNeuron = nextLayer.Neurons[j];
                        sumGradientTimesWeight += nextNeuron.Weights[i] * nextNeuron.Gradient;
                    }

                    neuron.Gradient = sumGradientTimesWeight * (neuron.Output * (1.0 - neuron.Output));
                }
            }
        }
        /// <summary> 
        /// Updates connection weights and biases across all hidden and output layers.
        /// </summary> 
        public void UpdateParameters(Layer[] layers, double learnRate)
        {
            for (int layerIndex = 1; layerIndex < layers.Length; layerIndex++)
            {
                Layer currentLayer = layers[layerIndex];
                Layer previousLayer = layers[layerIndex - 1];

                for (int i = 0; i < currentLayer.Neurons.Length; i++)
                {
                    Neuron neuron = currentLayer.Neurons[i];

                    for (int j = 0; j < previousLayer.Neurons.Length; j++)
                    {
                        neuron.Weights[j] -= learnRate * neuron.Gradient * previousLayer.Neurons[j].Output;
                    }

                    neuron.Bias -= learnRate * neuron.Gradient;
                }
            }
        }
        /// <summary> 
        /// Sigmoid activation function to map inputs into the range [0.0, 1.0]. 
        /// /// </summary>
        private double Activate(double weightedInput)
        {
            return 1.0 / (1.0 + Math.Exp(-weightedInput));
        }
    }
}