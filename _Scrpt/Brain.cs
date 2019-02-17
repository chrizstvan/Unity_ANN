using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Brain : MonoBehaviour
{

    ANN _artificialNeuralNet;
    double sumSquareError;

	// Use this for initialization
	void Start () 
    {
        _artificialNeuralNet = new ANN(2, 1, 1, 2, 0.8);

        List<double> result;

        // 1000 is epoch
        for (int i = 0; i < 10000; i++)
        {
            sumSquareError = 0;
            result = Train(1, 1, 1);
            sumSquareError += Mathf.Pow((float)result[0] - 1, 2);
            result = Train(1, 0, 0);
            sumSquareError += Mathf.Pow((float)result[0] - 0, 2);
            result = Train(0, 1, 0);
            sumSquareError += Mathf.Pow((float)result[0] - 0, 2);
            result = Train(0, 0, 1);
            sumSquareError += Mathf.Pow((float)result[0] - 1, 2);

        }
        Debug.Log("Sum square error : " + sumSquareError);

        result = Train(1, 1, 1);
        Debug.Log(" 1 1 " + result[0]);

        result = Train(1, 0, 0);
        Debug.Log(" 1 0 " + result[0]);

        result = Train(0, 1, 0);
        Debug.Log(" 0 1 " + result[0]);

        result = Train(0, 0, 1);
        Debug.Log(" 0 0 " + result[0]);

	}

    List<double> Train(double input_1, double input_2, double output)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();
        inputs.Add(input_1);
        inputs.Add(input_2);
        outputs.Add(output);

        return (_artificialNeuralNet.Go(inputs, outputs));
    }
	
	// Update is called once per frame
	void Update () {
		
	}
}
