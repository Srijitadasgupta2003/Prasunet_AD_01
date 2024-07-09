package Prasunet_AD_01;

import java.util.HashMap;
import java.util.Map;

public class CalorieEstimator {
    private static final Map<String, Integer> foodCalories = new HashMap<>();

    static {
        foodCalories.put("apple", 95);
        foodCalories.put("banana", 105);
        // Add more food items and their average calories
    }

    public static int estimateCalories(String foodItem) {
        return foodCalories.getOrDefault(foodItem, 0);
    }
}


