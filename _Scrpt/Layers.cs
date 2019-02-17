using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Layers
{

    public int numNeurons;
    public List<Neuron> neurons = new List<Neuron>();

    public Layers(int nNeurons, int numNeuronInput)
    {
        numNeurons = nNeurons;
        for (int i = 0; i < nNeurons; i++)
        {
            neurons.Add(new Neuron(numNeuronInput));
        }
    }
}
