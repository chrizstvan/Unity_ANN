using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ANN
{

    public int numInputs;
    public int numOutputs;
    public int numHidden;
    public int numPerHidden;
    public double alpha; // how fast your brain to learn

    List<Layers> layers = new List<Layers>();

    public ANN(int nI, int nO, int nH, int nPH, double a)
    {
        numInputs = nI;
        numOutputs = nO;
        numHidden = nH;
        numPerHidden = nPH;
        alpha = a;

        if (numHidden > 0)
        {
            layers.Add(new Layers(numPerHidden, numInputs));

            for (int i = 0; i < numHidden + 1; i++)
            {
                layers.Add(new Layers(numPerHidden, numPerHidden));
            }
            layers.Add(new Layers(numOutputs, numPerHidden));
        }
        else
        {
            layers.Add(new Layers(numOutputs, numInputs));
        }
    }

    //Method to train neural networks
    public List<double> Go(List<double> inputValue, List<double> desiredOutput)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();

        if (inputValue.Count != numInputs)
        {
            Debug.Log("ERROR: Number of input must be " + numInputs);
            return outputs;
        }

        inputs = new List<double>(inputValue);
        for (int i = 0; i < numHidden + 1; i++)
        {
            if (i > 0)
            {
                inputs = new List<double>(outputs); //jika bukan dari layer 0 input adalah output dr previous layer **
            }
            outputs.Clear();

            for (int j = 0; j < layers[i].numNeurons; j++)
            {
                double N = 0; // weight?
                layers[i].neurons[j].inputs.Clear();

                for (int k = 0; k < layers[i].neurons[j].numInput; k++)
                {
                    layers[i].neurons[j].inputs.Add(inputs[k]);
                    N += layers[i].neurons[j].weights[k] * inputs[k]; // weight x input
                }

                N -= layers[i].neurons[j].bias; //get bias
                //layers[i].neurons[j].output = ActivationFunction(N);
                outputs.Add(layers[i].neurons[j].output); //output ini nanti balik lagi ke **
            }
        }

        //UpdateWeights(outputs, desiredOutput);
        return outputs;
    }

}
