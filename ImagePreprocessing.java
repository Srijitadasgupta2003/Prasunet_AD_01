import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;

import java.io.File;

public class ImagePreprocessing {
    private static final int IMG_WIDTH = 224;
    private static final int IMG_HEIGHT = 224;

    public static Mat preprocessImage(String imagePath) {
        Mat image = opencv_imgcodecs.imread(imagePath);
        Mat resizedImage = new Mat();
        opencv_imgproc.resize(image, resizedImage, new Size(IMG_WIDTH, IMG_HEIGHT));
        return resizedImage;
    }

    public static void main(String[] args) {
        File folder = new File("path/to/your/dataset");
        for (File file : folder.listFiles()) {
            Mat processedImage = preprocessImage(file.getAbsolutePath());
            // Save or use the processed image for further steps
        }
    }
}

