using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ANN
{

    public int numInputs;
    public int numOutputs;
    public int numHidden;
    public int numPerHidden;
    public double alpha; // how fast your brain to learn (learning rate)

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

            for (int i = 0; i < numHidden - 1; i++)
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
                layers[i].neurons[j].output = ActivationFunction(N);
                outputs.Add(layers[i].neurons[j].output); //output ini nanti balik lagi ke **
            }
        }

        UpdateWeights(outputs, desiredOutput);
        return outputs;
    }

    void UpdateWeights(List<double> outputs, List<double> desiredOutput)
    {
        double error;
        for (int i = numHidden; i >= 0; i--) //first layer -- loop backward
        {
            for (int j = 0; j < layers[i].numNeurons; j++) //first layer first neuron
            {
                if (i == numHidden) //where we in the end (output layer)
                {
                    error = desiredOutput[j] - outputs[j];
                    layers[i].neurons[j].errorGradient = outputs[j] * (1 - outputs[j]) * error; //rumus dapet di wikipedia error gradient(Delta Rule)
                }
                else
                {
                    layers[i].neurons[j].errorGradient = layers[i].neurons[j].output * (1 - layers[i].neurons[j].output);
                    double errorGradSum = 0;
                    for (int p = 0; p < layers[i + 1].numNeurons; p++)
                    {
                        errorGradSum += layers[i + 1].neurons[p].errorGradient * layers[i + 1].neurons[p].weights[j];
                    }
                    layers[i].neurons[j].errorGradient *= errorGradSum;
                }
                for (int k = 0; k < layers[i].neurons[j].numInput; k++)
                {
                    if (i == numInputs)
                    {
                        error = desiredOutput[j] - outputs[j];
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * error; 
                    }
                    else
                    {
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].errorGradient;
                    }
                    // define bias
                    layers[i].neurons[j].bias += alpha * layers[i].neurons[j].errorGradient * -1;
                }
            }
        }
    }

    double ActivationFunction(double value)
    {
        return Sigmoid(value);
    }

    double Step(double value) // binary step
    {
        if (value < 0)
        {
            return 0;
        }
        else
        {
            return 1;
        }
    }

    double Sigmoid(double value) //logistic softstep
    {
        double k = (double)System.Math.Exp(value);
        return k / (1.0f + k);
    }
}
