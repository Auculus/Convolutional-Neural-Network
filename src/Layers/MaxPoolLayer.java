package Layers;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolLayer extends Layer{

    private int _stepSize;
    private int _windowSize;

    private int _inputLength;
    private int _inputRows;
    private int _inputColumns;

    List<int[][]> _lastMaxRow;
    List<int[][]> _lastMaxColumn;

    public MaxPoolLayer(int _stepSize, int _windowSize, int _inputLength, int _inputRows, int _inputColumns) {
        this._stepSize = _stepSize;
        this._windowSize = _windowSize;
        this._inputLength = _inputLength;
        this._inputRows = _inputRows;
        this._inputColumns = _inputColumns;
    }

    public List<double[][]> maxPoolForwardPass(List<double[][]> input){

        List<double[][]> output = new ArrayList<>();
        _lastMaxRow = new ArrayList<>();
        _lastMaxColumn = new ArrayList<>();


        for(int l = 0; l<input.size(); l++){
            output.add(pool(input.get(l)));
        }
        return output;
    }

    public double[][] pool(double[][] input){

        double[][] output = new double[getOutputRows()][getOutputCols()];

        int[][] maxRows = new int[getOutputRows()][getOutputCols()];
        int[][] maxColumns = new int[getOutputRows()][getOutputCols()];

        for (int r = 0; r<getOutputRows(); r+=_stepSize){
            for (int c=0; c< getOutputCols(); c+=_stepSize){

                double max = 0.0;
                maxRows[r][c] = -1;
                maxColumns[r][c] = -1;

                for(int x = 0; x<_windowSize; x++){
                    for(int y =0 ; y<_windowSize; y++){
                        if (max< input[r+ x][c+y]){
                            max = input[r+ x][c+y];
                            maxRows[r][c] = r+x;
                            maxColumns[r][c] = c+y;
                        }
                    }
                }

                output[r][c] = max;
            }
        }

        _lastMaxRow.add(maxRows);
        _lastMaxColumn.add(maxColumns);

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> outputPool = maxPoolForwardPass(input);
        return _nextLayer.getOutput(outputPool);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrix = vectorToMatrix(input, getOutputLength(),_inputRows,_inputColumns);
        return getOutput(matrix);
    }

    @Override
    public void backPropagation(double[] dLdO) {
        List<double[][]> matrix = vectorToMatrix(dLdO, getOutputLength(),getOutputRows(),getOutputCols());
        backPropagation(matrix);
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        List<double[][]> dXdL = new ArrayList<>();

        int l = 0;
        for (double[][] array: dLdO){
            double[][] error = new double[_inputRows][_inputColumns];

            for (int r = 0; r<getOutputRows(); r++){
                for (int c = 0; c<getOutputCols(); c++){
                    int max_i = _lastMaxRow.get(l)[r][c];
                    int max_j = _lastMaxColumn.get(l)[r][c];

                    if (max_i != -1){
                        error[max_i][max_j] += array[r][c];
                    }
                }
            }
            dXdL.add(error);
            l++;
        }

        if (_previousLayer != null){
            _previousLayer.backPropagation(dXdL);
        }
    }

    @Override
    public int getOutputLength() {
        return _inputLength;
    }

    @Override
    public int getOutputRows() {
        return (_inputRows -_windowSize)/_stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (_inputColumns -_windowSize)/_stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return _inputLength*getOutputCols()*getOutputRows();
    }
}
