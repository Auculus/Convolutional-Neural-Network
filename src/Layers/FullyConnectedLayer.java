package Layers;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer{
    private double[][] _weights;
    private long SEED;
    private final double leak = 0.01;
    private final double learningRate;

    public FullyConnectedLayer(int _intLength, int _outLength, long SEED, double learning_rate) {
        this._intLength = _intLength;
        this._outLength = _outLength;
        this.learningRate = learning_rate;
        this.SEED = SEED;

        _weights = new double[this._intLength][this._outLength];
        setRandomWeights();
    }

    private final int _intLength;
    private final int _outLength;

    private double[] lastZ;
    private double[] lastX;

    public double[] fullyConnectedForwardPass(double[] input){
        if (input.length!= _intLength) return null;

        double[] z = new double[_outLength];
        double[] out = new double[_outLength];

        lastX = input;

        for (int i = 0 ; i<_intLength; i++){
            for(int j = 0; j<_outLength; j++){
                z[j] += input[i]*_weights[i][j];
            }
        }
        lastZ = z;

        if (_nextLayer != null) {
            for (int i = 0; i < _intLength; i++) {
                for (int j = 0; j < _outLength; j++) {
                    out[j] = reLU(z[j]);
                }
            }
        } else {
            double[] exp_z = new double[_outLength];
            for (int j =0; j<_outLength; j++){
                exp_z[j] = Math.exp(z[j]);
            }
            for ( int k = 0; k<_outLength; k++){
                out[k] = exp_z[k]/ summation(exp_z); // softmax for each output
            }
        }

        return out;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return getOutput(vector);

    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = fullyConnectedForwardPass(input);
        if (_nextLayer != null){
            return _nextLayer.getOutput(forwardPass);
        } else {
            return forwardPass;
        }
    }

    @Override
    public void backPropagation(double[] dLdO) {

        double[] dLdx = new double[_intLength];

        double dOdz;
        double dzdw;
        double dLdw;
        double dzdx;

        for (int k = 0; k<_intLength; k++){

            double dLdx_sum = 0;

            for (int j = 0; j<_outLength; j++){

                if (_nextLayer != null){
                    dOdz = reLU_derivative(lastZ[j]);

                    dzdw = lastX[k];
                    dzdx  = _weights[k][j];

                    dLdw = dLdO[j]*dOdz *dzdw;

                    _weights[k][j] -= dLdw * learningRate;

                    dLdx_sum += dLdO[j] *dOdz *dzdx;
                } else {
                    /*double kronecker_del = 0;
                    if (k == j){
                        kronecker_del = 1;
                    }
                    dOdz = lastZ[j] * (kronecker_del - lastZ[k]);*/
                    dOdz = dLdO[j];
                    dzdw = lastX[k];
                    dzdx  = _weights[k][j];

                    dLdw = dOdz * dzdw;

                    _weights[k][j] -= dLdw *learningRate;

                    dLdx_sum += dOdz* dzdx;
                }

            }

            dLdx[k] = dLdx_sum;
        }

        if (_previousLayer != null){
            _previousLayer.backPropagation(dLdx);
        }
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        double[] vector = matrixToVector(dLdO);
        backPropagation(vector);

    }

    @Override
    public int getOutputLength() {
        return 0;
    }

    @Override
    public int getOutputRows() {
        return 0;
    }

    @Override
    public int getOutputCols() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return _outLength;
    }

    public double reLU(double x){
        if (x<=0) return 0d;
        else return x;
    }

    public double reLU_derivative(double x){
        if (x<=0) return leak;
        else return 1d;
    }

    public double summation(double[] array){
        double sum = 0;
        for (double i : array){
            sum+=i;
        }
        return sum;
    }

    public void setRandomWeights(){
        Random random_gen = new Random(SEED);

        for (int i = 0; i < _intLength;i++){
            for (int j= 0 ; j< _outLength; j++){
                _weights[i][j] = random_gen.nextGaussian();
            }
        }
    }
}
