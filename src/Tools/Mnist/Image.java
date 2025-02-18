package Tools.Mnist;

public class Image {

    private final double[][] data;
    private final int label;

    public Image(double[][] dat, int label_val){
        this.label = label_val;
        this.data = dat;
    }

    public int getLabel(){
        return label;
    }

    public double[][] getData() {
        return data;
    }

    @Override
    public String toString(){
        String s = label + ", \n";
        for (int i = 0; i< data.length; i++){
            for (int j = 0; j < data[0].length; j++){
                s += data[i][j] +", ";
            }
            s+="\n";
        }
        return s;
    }
}
