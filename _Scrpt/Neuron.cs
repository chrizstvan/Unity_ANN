﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Neuron
{

    public int numInput;
    public double bias;
    public double output;
    public double errorGradient;
    public List<double> weights = new List<double>();
    public List<double> inputs = new List<double>();

    public Neuron(int nInputs)
    {
        bias = Random.Range(-1, 1);
        numInput = nInputs;
        for (int i = 0; i < nInputs; i++)
        {
            weights.Add(Random.Range(-1, 1));
        }
    }
}
