package Layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static Tools.MatrixUtil.add;
import static Tools.MatrixUtil.scalarMultiply;

public class ConvolutionLayer extends Layer{

    private long seed;

    private List<double[][]> filters;
    private int filter_size;
    private int step_size;

    private int _intLength; // expected input
    private int _intRows;
    private int _intCols;
    private double _learningRate;

    private List<double[][]> _lastInput;

    public ConvolutionLayer(int filter_size, int step_size, int _intLength, int _intRows, int _intCols, long seed, double learningRate ,int noFilters) {
        this.filter_size = filter_size;
        this.step_size = step_size;
        this._intLength = _intLength;
        this._intRows = _intRows;
        this._intCols = _intCols;
        this.seed = seed;
        _learningRate = learningRate;

        genFilters(noFilters);
    }

    public void genFilters(int noFilters){
        List<double[][]> filters_output = new ArrayList<>();
        Random random = new Random(seed);

        for( int i = 0; i<noFilters; i++){
            double[][] mat = new double[filter_size][filter_size];

            for ( int j =0 ; j<filter_size;j++){
                for ( int k = 0; k<filter_size; k++){
                    double value = random.nextGaussian();
                    mat[j][k] = value;
                }
            }
            filters_output.add(mat);
        }
        filters = filters_output;

    }

    public List<double[][]> convolutionForwardPass(List<double[][]> list){
        _lastInput = list;

        List<double[][]> output = new ArrayList<>();
        for (int m = 0; m<list.size(); m++){
            for (double[][] filter : filters){
                output.add(convolve(list.get(m), filter, step_size));
            }
        }
        return output;
    }

    public double[][] convolve(double[][] input, double[][] filter, int step_size){

        int outRows = (input.length - filter.length)/step_size +1;
        int outColumns = (input[0].length - filter[0].length) /step_size + 1;

        int inRows = input.length;
        int inColumns = input[0].length;

        int fRows = filter.length;
        int fColumns = filter[0].length;

        double[][] output = new double[outRows][outColumns];
        int outRow = 0;
        int outCol;

        for (int i = 0; i<= inRows - fRows; i+=step_size ){

            outCol = 0;

            for ( int j = 0 ; j<= inColumns - fColumns; j+=step_size){
                double sum = 0;

                for ( int x= 0; x<fRows; x++){ // To move the positioning of the filter
                    for (int y = 0 ; y<fColumns; y++){
                        int inputRowIndex = i+x;
                        int inputColumnIndex = j+y;

                        double value = filter[x][y] * input[inputRowIndex][inputColumnIndex];
                        sum += value;
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;
            }
            outRow++;
        }
        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> output = convolutionForwardPass(input);
        return _nextLayer.getOutput(output);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> mInput = vectorToMatrix(input, _intLength, _intRows, _intCols);
        return getOutput(mInput);
    }

    @Override
    public void backPropagation(double[] dLdO) {
        List<double[][]> dLd0Matrix = vectorToMatrix(dLdO, _intLength, _intRows, _intCols);
        backPropagation(dLd0Matrix);
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {

        List<double[][]> filtersDelta = new ArrayList<>();
        List<double[][]> dLd0PreviousLayer = new ArrayList<>();

        for (int f = 0; f <filters.size(); f++){
            filtersDelta.add(new double[filter_size][filter_size]);
        }

        for (int inp = 0; inp<_lastInput.size(); inp++){

            double[][] errorForInput = new double[_intRows][_intCols];

            for (int filter = 0; filter<filters.size();filter++){

                double[][] curr_filter = filters.get(filter);
                double[][] curr_error = dLdO.get(inp*filters.size() + filter);

                double[][] curr_error_spaced = spaceArray(curr_error);
                double[][] dLdF = convolve(_lastInput.get(inp), curr_error_spaced, 1);

                double[][] delta = scalarMultiply(-1.0*_learningRate, dLdF);

                double[][] newTotalData = add(filtersDelta.get(filter), delta);
                filtersDelta.set(filter, newTotalData);

                double[][] flippedError = flipHorizontally(flipVertically(curr_error_spaced));
                errorForInput = add(errorForInput,fullConvolve(curr_filter, flippedError));

            }

            dLd0PreviousLayer.add(errorForInput);

        }

        for (int filter = 0; filter<filters.size();filter++){
            double[][] modified = add(filtersDelta.get(filter), filters.get(filter));
            filters.set(filter, modified);
        }

        if (_previousLayer != null){
            _previousLayer.backPropagation(dLd0PreviousLayer);
        }
    }

    public double[][] spaceArray(double[][] loss_wrt_output_matrix){
        if (step_size == 1) {
            return loss_wrt_output_matrix;
        }

        int outRows = (loss_wrt_output_matrix.length -1) * step_size +1; // Same as input-filterSize
        int outCols = (loss_wrt_output_matrix[0].length -1) *step_size + 1;

        double[][] output =  new double[outRows][outCols];

        for (int row = 0; row < loss_wrt_output_matrix.length; row++){
            for (int col = 0; col< loss_wrt_output_matrix[0].length; col++){
                output[row*step_size][col*step_size] = loss_wrt_output_matrix[row][col];
            }
        }

        return output;
    }

    public double[][] flipHorizontally(double[][] array){
        int rows = array.length;
        int cols = array[0].length;

        double[][] output = new double[rows][cols];

        for (int row = 0; row< rows; row++){
            for (int col = 0; col<cols; col++){
                output[rows - row -1][col] = array[row][col];
            }
        }

        return output;
    }

    public double[][] flipVertically(double[][] array){
        int rows = array.length;
        int cols = array[0].length;

        double[][] output = new double[rows][cols];

        for (int row = 0; row<rows; row++){
            for (int col = 0; col<cols; col++){
                output[row][cols -col -1] = array[row][col];
            }
        }

        return output;
    }

    public double[][] fullConvolve(double[][] input, double[][] filter){
        int outRows = (input.length + filter.length) +1;
        int outColumns = (input[0].length + filter[0].length) + 1;

        int inRows = input.length;
        int inColumns = input[0].length;

        int fRows = filter.length;
        int fColumns = filter[0].length;

        double[][] output = new double[outRows][outColumns];
        int outRow = 0;
        int outCol;

        for (int i = -fRows + 1; i< inRows; i++ ){

            outCol = 0;
            for ( int j = -fColumns + 1 ; j< inColumns ;j++){
                double sum = 0.0;

                for ( int x= 0; x<fRows; x++){ // To move the positioning of the filter
                    for (int y = 0 ; y<fColumns; y++){
                        int inputRowIndex = i+x;
                        int inputColIndex = j+y;

                        if(inputRowIndex >=0 && inputColIndex>=0 && inputRowIndex < inRows && inputColIndex <inColumns){
                            double value = filter[x][y] * input[inputRowIndex][inputColIndex];
                            sum += value;
                        }
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;
            }
            outRow++;
        }
        return output;
    }

    @Override
    public int getOutputLength() {
        return filters.size() * _intLength;
    }

    @Override
    public int getOutputRows() {
        return (_intRows - filter_size)/step_size +1;
    }

    @Override
    public int getOutputCols() {
        return (_intCols - filter_size)/step_size +1;
    }

    @Override
    public int getOutputElements() {
        return getOutputCols()*getOutputRows()*getOutputLength();
    }
}
