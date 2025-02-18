import Network.NetworkBuilder;
import Network.NeuralNetwork;
import Tools.Mnist.DataReader;
import Tools.Mnist.Image;

import java.util.Arrays;
import java.util.List;

import static java.util.Collections.shuffle;

public class Main {
    public static void main(String[] args){

        long SEED = 123;

        System.out.println(" Starting data loading... ");

        List<Image> imagesTest = new DataReader().readData("src/mnist_fashion_test.csv");
        List<Image> imagesTrain = new DataReader().readData("src/mnist_fashion_train.csv");

        System.out.println("Images Train size : " + imagesTrain.size());
        System.out.println("Images Test size : " + imagesTest.size());

        NetworkBuilder builder = new NetworkBuilder(28,28, 256*100);
        builder.addConvolutionLayer(10, 5, 1, 0.1, SEED);
        builder.addMaxPoolLayer(3, 2);
        builder.addFullyConnectedLayer(10, 0.1, SEED);

        NeuralNetwork net = builder.build();

        float rate = net.test(imagesTest);
        System.out.println("Pre Training Success rate: " + rate);
        System.out.println("Confusion Matrix is ");
        net.displayConfusionMatrix();

        int epochs = 3;

        for (int i = 0; i<epochs; i++){
            shuffle(imagesTrain);
            net.train(imagesTrain);

            rate = net.test(imagesTest);
            System.out.println("Success Rate after round " + i + " is : " + rate);
            System.out.println("Confusion Matrix is: ");
            net.displayConfusionMatrix();
        }

    }
}
