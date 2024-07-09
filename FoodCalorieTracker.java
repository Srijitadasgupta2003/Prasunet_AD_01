package Prasunet_AD_01;

import org.bytedeco.opencv.opencv_core.Mat;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class FoodCalorieTracker {
    private MultiLayerNetwork model;

    public FoodCalorieTracker(MultiLayerNetwork model) {
        this.model = model;
    }

    public String recognizeFood(Mat image) {
        INDArray input = Nd4j.create(image.data().asFloat(), new int[]{1, 3, 224, 224});
        INDArray output = model.output(input);
        int foodIndex = Nd4j.argMax(output, 1).getInt(0);
        // Map foodIndex to food label
        return "apple"; // Placeholder, map the index to actual label
    }

    public static void main(String[] args) {
        FoodCalorieTracker tracker = new FoodCalorieTracker(model);
        Mat image = ImagePreprocessing.preprocessImage("path/to/food/image.jpg");
        String foodItem = tracker.recognizeFood(image);
        int calories = CalorieEstimator.estimateCalories(foodItem);
        System.out.println("Food: " + foodItem + ", Calories: " + calories);
    }
}


