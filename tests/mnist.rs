#[cfg(test)]
mod tests {

    #[cfg(test)]
    mod mnist {

        use mnist::*;
        use ndarray::prelude::*;
        use std::io::{self, Write};

        use rustgrad::autograd::nonlinearity::NonLinearity::*;
        use rustgrad::structures::layer::LayerSpec::*;
        use rustgrad::structures::loss::{Loss, LossSpec};
        use rustgrad::structures::mlp::MLP;

        #[test]
        #[ignore]
        // To run this test use command:
        // [cargo test mnist_full_scale --release -- --nocapture --ignored]
        fn mnist_full_scale() {
            // TODO: remove some duplication
            const TRAIN_ITER: i32 = 5_000; // 20 sec per 1k

            let train = 45_000;
            let dev = 5_000;
            let test = 10_000;

            let one_hot = |x: f64| -> Vec<f64> {
                let mut v = vec![0.0; 10];
                v[x as usize] = 1.0;
                v
            };

            // Deconstruct the returned Mnist struct.
            println!("Loading MNIST dataset...");
            let Mnist {
                trn_img,
                trn_lbl,
                tst_img,
                tst_lbl,
                ..
            } = MnistBuilder::new()
                .base_path("tests/input/MNIST")
                .label_format_digit()
                .training_set_length(50_000)
                .validation_set_length(10_000)
                .test_set_length(10_000)
                .finalize();

            // Can use an Array2 or Array3 here (Array3 for visualization)
            let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
                .expect("Error converting images to Array3 struct")
                .map(|x| *x as f64 / 256.0);

            // Convert the returned Mnist struct to Array2 format
            let train_labels: Array2<f64> = Array2::from_shape_vec((50_000, 1), trn_lbl)
                .expect("Error converting training labels to Array2 struct")
                .map(|x| *x as f64);

            // ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
            //                          Initialize and train the model                         //
            // ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //

            // Prepare the training data
            let x_train = (0..train)
                .map(|image_num| {
                    train_data
                        .slice(s![image_num, .., ..])
                        .iter()
                        .map(|x| *x as f64)
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>();

            let y_train = (0..train)
                .map(|image_num| {
                    train_labels
                        .slice(s![image_num, ..])
                        .iter()
                        .flat_map(|x| one_hot(*x))
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>();

            println!("Constructing the model...");
            let mlp = MLP::new(784)
                .add_layer(FullyConnected(22))
                .add_layer(NonLinear(ReLu))
                .add_layer(FullyConnected(10))
                .build();
            let loss = Loss::new(&mlp).add_loss(LossSpec::MultiHinge).build();

            println!("Training the model...");
            for i in 0..TRAIN_ITER {
                print!(
                    "\rLoss = {:.8}, \t Iter = {}",
                    loss.rand_batch_train(&mlp, &x_train, &y_train, 64, 0.2),
                    i
                );
                io::stdout().flush().unwrap();
            }
            println!();

            // ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
            //                              Evaluate on dev data                               //
            // ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
            // Prepare dev data
            let x_dev = (train..train + dev)
                .map(|image_num| {
                    train_data
                        .slice(s![image_num, .., ..])
                        .iter()
                        .map(|x| *x as f64)
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>();

            let y_dev = (train..train + dev)
                .map(|image_num| {
                    train_labels
                        .slice(s![image_num, ..])
                        .iter()
                        .flat_map(|x| one_hot(*x))
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>();

            println!("--------------------------------Accuracy on dev data--------------------------------");
            let right_predictions = (0..dev).map(|i| {
                let prediction: usize = mlp
                    .eval(&x_dev[i])
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                let label: usize = y_dev[i]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                (prediction == label) as i32
            });
            println!(
                "Accuracy on dev: {}",
                right_predictions.sum::<i32>() as f64 / dev as f64
            );

            // ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
            //                              Run on the test data                               //
            // ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //

            let test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
                .expect("Error converting images to Array3 struct")
                .map(|x| *x as f64 / 256.);

            let test_labels: Array2<f64> = Array2::from_shape_vec((10_000, 1), tst_lbl)
                .expect("Error converting testing labels to Array2 struct")
                .map(|x| *x as f64);

            let x_test = (0..test)
                .map(|image_num| {
                    test_data
                        .slice(s![image_num, .., ..])
                        .iter()
                        .map(|x| *x as f64)
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>();

            let y_test = (0..test)
                .map(|image_num| {
                    test_labels
                        .slice(s![image_num, ..])
                        .iter()
                        .flat_map(|x| one_hot(*x))
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>();

            println!("-------------------------------Accuracy on test data--------------------------------");
            let right_predictions = (0..test).map(|i| {
                let prediction: usize = mlp
                    .eval(&x_test[i])
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                let label: usize = y_test[i]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                (prediction == label) as i32
            });
            println!(
                "Accuracy on test: {}",
                right_predictions.sum::<i32>() as f64 / test as f64
            );
        }
    }
}
