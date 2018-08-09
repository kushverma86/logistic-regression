package com.PersonalProject;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.*;

public class LogisticNew {
    public static void main(String[] args) throws Exception {
        CSVLoader source=new CSVLoader();
        source.setSource(new File("ex2data1.csv"));
        Instances data= source.getDataSet();
        data.setClassIndex(data.numAttributes()-1);
        NumericToNominal filter=new NumericToNominal();
        filter.setInputFormat(data);
        data=Filter.useFilter(data,filter);
        System.out.println(data);

        Logistic lg=new Logistic();

        lg.setMaxIts(400);
        lg.buildClassifier(data);




      //  CSVLoader source2=new CSVLoader();
        source.setSource(new File("ex2Test.csv"));
        Instances unlabeled= source.getDataSet();

//        Instances unlabeled =new Instances(new BufferedReader(new FileReader("ex2Test.csv")));
        unlabeled.setClassIndex(unlabeled.numAttributes()-1);
        double clsLabel=lg.classifyInstance(unlabeled.instance(1));

        Instances labeled = new Instances(unlabeled);

        // label instances
        for (int i = 0; i < unlabeled.numInstances(); i++) {
      //      double clsLabel = lg.classifyInstance(unlabeled.instance(i));
            System.out.println(clsLabel);
            labeled.instance(i).setClassValue(clsLabel);
        }
        // save labeled data
        BufferedWriter writer = new BufferedWriter(
                new FileWriter("result.csv"));
        writer.write(labeled.toString());
        writer.newLine();
        writer.flush();
        writer.close();



    }






}
