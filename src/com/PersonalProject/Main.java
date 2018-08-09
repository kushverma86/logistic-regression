package com.PersonalProject;

import java.util.ArrayList;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        System.out.println("Enter the file name to load data :");
        CsvParser csvp=new CsvParser();               // parsing Csv File
        ArrayList<Instance> Instances= csvp.getdata();                               // get the instances arraylist
        int _ndims=csvp.getAttCount();
        System.out.println(_ndims);
        int _nrows=Instances.size();
        System.out.println(_nrows);
        InputOutputVectors IOV=new InputOutputVectors(Instances,_ndims);
        double[][] featureMatrix=IOV.getX_inputs();
        double[] outputVecor=IOV.getY_outputs();
        ArrayList<ArrayList<Double>> X=new ArrayList<>();
        ArrayList<Double> Y=new ArrayList<>();

        for (int i=0;i<featureMatrix.length;i++)
        {
            ArrayList<Double> inputRow=new ArrayList<Double>();
            for (int j=0;j<featureMatrix[i].length;j++)
            {
                inputRow.add(featureMatrix[i][j]);
            }
            X.add(inputRow);
        }

        for (int i=0;i<outputVecor.length;i++)
        {
            Y.add(outputVecor[i]);
        }

        Logistic lr=new Logistic(X,Y);
        lr.init();
        lr.train(0.001,8000000);

        Scanner sc=new Scanner(System.in);
        double[] testData=new double[X.get(0).size()];
        testData[0]=1;

        for (int i=1;i<X.get(0).size();i++)
        {
            testData[i]=sc.nextDouble();
        }
       // double[] theta={-24.932982, 0.204408, 0.199618};

        double result=lr.predict(testData);

        System.out.println("Result: "+result);
    }




}
