use xgboost_rust::{Booster, XGBoostResult};

fn main() -> XGBoostResult<()> {
    println!("XGBoost Rust Bindings - Basic Usage Example");
    println!("============================================\n");

    // Note: This example assumes you have a trained XGBoost model file.
    // To create one, you can use Python:
    //
    // ```python
    // import xgboost as xgb
    // from sklearn.datasets import load_iris
    // from sklearn.model_selection import train_test_split
    //
    // # Load iris dataset
    // iris = load_iris()
    // X_train, X_test, y_train, y_test = train_test_split(
    //     iris.data, iris.target, test_size=0.2, random_state=42
    // )
    //
    // # Train model
    // dtrain = xgb.DMatrix(X_train, label=y_train)
    // params = {
    //     'objective': 'multi:softprob',
    //     'num_class': 3,
    //     'max_depth': 3,
    //     'eta': 0.3
    // }
    // bst = xgb.train(params, dtrain, num_boost_round=10)
    //
    // # Save model
    // bst.save_model('iris_model.json')
    // ```

    // Load a pre-trained model
    let model_path = "iris_model.json";

    println!("Loading model from: {}", model_path);
    let booster = match Booster::load(model_path) {
        Ok(b) => {
            println!("✓ Model loaded successfully\n");
            b
        }
        Err(e) => {
            eprintln!("✗ Failed to load model: {}", e);
            eprintln!("\nPlease create a model file first using the Python code in the example.");
            return Err(e);
        }
    };

    // Get model information
    let num_features = booster.num_features()?;
    println!("Model expects {} features\n", num_features);

    // Example prediction data (4 features for iris dataset)
    // This is a sample from the iris dataset
    let data = vec![
        5.1, 3.5, 1.4, 0.2, // Row 1: Setosa
        6.7, 3.0, 5.2, 2.3, // Row 2: Virginica
        5.9, 3.0, 4.2, 1.5, // Row 3: Versicolor
    ];
    let num_rows = 3;
    let num_features = 4;

    println!("Making predictions on {} samples...", num_rows);

    // Make predictions
    let predictions = booster.predict(&data, num_rows, num_features, 0, false)?;

    println!("✓ Predictions complete\n");

    // Print results
    println!("Predictions (probabilities for 3 classes):");
    for i in 0..num_rows {
        println!("Sample {}:", i + 1);
        println!("  Class 0 (Setosa):     {:.4}", predictions[i * 3]);
        println!("  Class 1 (Versicolor): {:.4}", predictions[i * 3 + 1]);
        println!("  Class 2 (Virginica):  {:.4}", predictions[i * 3 + 2]);

        // Find predicted class (argmax)
        let mut max_prob = predictions[i * 3];
        let mut predicted_class = 0;
        for j in 1..3 {
            if predictions[i * 3 + j] > max_prob {
                max_prob = predictions[i * 3 + j];
                predicted_class = j;
            }
        }
        println!("  → Predicted class: {}\n", predicted_class);
    }

    println!("Example completed successfully!");

    Ok(())
}
