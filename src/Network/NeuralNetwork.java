package Network;

import Layers.Layer;
import Tools.Mnist.Image;

import java.util.ArrayList;
import java.util.List;

import static Tools.MatrixUtil.add;
import static Tools.MatrixUtil.scalarMultiply;

public class NeuralNetwork {
    List<Layer> _layers;
    double scalingFactor;
    double[][] confusionMatrix = new double[10][10];

    public NeuralNetwork(List<Layer> _layers, double scaleFactor) {
        this._layers = _layers;
        this.scalingFactor = scaleFactor;
        linkLayers();
    }


    public void linkLayers(){
        if(_layers.size() <=1)return;

        for (int i = 0; i<_layers.size(); i++){
            if (i == 0){
                _layers.get(i).set_nextLayer(_layers.get(i+1));
            }
            else if (i == (_layers.size() -1)){
                _layers.get(i).set_previousLayer(_layers.get(i - 1));
            }
            else{
                _layers.get(i).set_previousLayer(_layers.get(i-1));
                _layers.get(i).set_nextLayer(_layers.get(i+1));
            }
        }
    }

    public double[] getErrors(double[] networkOutput, int correctOutput){
        int numClasses = networkOutput.length;

        double[] expectedOutput  = new double[numClasses];

        expectedOutput[correctOutput] =1;

        return add(networkOutput, scalarMultiply(-1, expectedOutput));  // cross entropy loss or regular error loss
    }

    public int getMaxIndex(double[] in){
        double max = 0;
        int index = 0;

        for (int i= 0; i<in.length; i++){
            if(in[i]> max){
                max = in[i];
                index = i;
            }
        }

        return index;
    }

    public int guess(Image image){
        List<double[][]> inList = new ArrayList<>();
        inList.add(scalarMultiply((1.0/scalingFactor),image.getData()));

        double[] out = _layers.get(0).getOutput(inList);
        //System.out.println(Arrays.toString(out));
        int guess = getMaxIndex(out);

        return guess;
    }

    public float test (List<Image> images){
        resetConfusionMatrix();
        int correct = 0;
        for (Image img: images){
            int gs = guess(img);

            if(gs == img.getLabel()){
                correct ++;
            }
            confusionMatrixUpdate(img, gs);

        }

        return ((float)correct/(float)images.size());
    }

    public void confusionMatrixUpdate(Image img, int output_val){
        int expected_out = img.getLabel();

        confusionMatrix[expected_out][output_val] += 1;
    }

    public void resetConfusionMatrix(){
        confusionMatrix = new double[10][10];
    }

    public void displayConfusionMatrix(){
        for (int i = 0; i<confusionMatrix.length; i++){
            String s = "| ";
            for (int j = 0; j<confusionMatrix[0].length; j++) {
                s += ((int)confusionMatrix[i][j] + " ");
            }
            System.out.println(s + "|");
        }
    }

    public void train(List<Image> images){

        for (Image img: images){
            List<double[][]> inList = new ArrayList<>();
            inList.add(scalarMultiply((1.0/scalingFactor),img.getData()));

            double[] out = _layers.get(0).getOutput(inList);
            //System.out.println(Arrays.toString(out));
            double[] dld0 = getErrors(out, img.getLabel());

            _layers.get(_layers.size() - 1).backPropagation(dld0);

        }
    }
}
