package Network;

import Layers.ConvolutionLayer;
import Layers.FullyConnectedLayer;
import Layers.Layer;
import Layers.MaxPoolLayer;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {

    private NeuralNetwork net;
    private int _inputRows;
    private int _inputCols;
    private double _scaleFactor;
    List<Layer> _layers;

    public NetworkBuilder(int _inputRows, int _inputCols, double scaleFactor) {
        this._inputRows = _inputRows;
        this._inputCols = _inputCols;
        this._scaleFactor = scaleFactor;

        _layers = new ArrayList<>();
    }

    public void addConvolutionLayer(int numFilters, int filterSize, int stepSize, double learningRate, long seed){
        if (_layers.isEmpty()){
            _layers.add(new ConvolutionLayer(filterSize, stepSize, 1, _inputRows,_inputCols,seed, learningRate, numFilters));
        } else{
            Layer prev_layer = _layers.get(_layers.size() -1);
            _layers.add(new ConvolutionLayer(filterSize, stepSize, prev_layer.getOutputLength(), prev_layer.getOutputRows(),prev_layer.getOutputCols() ,seed, learningRate, numFilters));
        }
    }

    public void addMaxPoolLayer(int windowSize, int stepSize){
        if (_layers.isEmpty()){
            _layers.add(new MaxPoolLayer(stepSize,windowSize,1, _inputRows, _inputCols));
        } else{
            Layer prev_layer = _layers.get(_layers.size() -1);
            _layers.add(new MaxPoolLayer(stepSize,windowSize, prev_layer.getOutputLength(), prev_layer.getOutputRows(), prev_layer.getOutputCols()));
        }
    }

    public void addFullyConnectedLayer(int outLength, double learningRate, long SEED){
        if (_layers.isEmpty()){
            _layers.add(new FullyConnectedLayer(_inputCols*_inputRows,outLength,SEED,learningRate));
        } else{
            Layer prev_layer = _layers.get(_layers.size() -1);
            _layers.add(new FullyConnectedLayer(prev_layer.getOutputElements(),outLength, SEED, learningRate));
        }
    }

    public NeuralNetwork build(){
        net = new NeuralNetwork(_layers, _scaleFactor);
        return net;
    }
}
