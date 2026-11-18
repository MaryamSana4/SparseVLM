            )

        pd.DataFrame(history).to_csv(
            RESULTS_DIR / "training_history.csv",
            index=False,
        )

    total_training_time = time.perf_counter() - training_start

    print("\nTraining completed.")
    print(f"Total training time: {total_training_time:.2f} seconds")

    # --------------------------------------------------------
    # Load best model for final testing
    # --------------------------------------------------------
    best_path = CHECKPOINT_DIR / "best_model.pth"

    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location=DEVICE)
        model.classifier.load_state_dict(best_checkpoint["classifier_state_dict"])
        print(
            "Loaded best checkpoint from epoch:",
            best_checkpoint["epoch"],
        )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------
    print("\nEvaluating on official test split...")

    test_result = run_epoch(
        model,
        test_loader,
        criterion,
        optimizer=None,
    )

    metrics, cm = calculate_metrics(
        test_result["labels"],
        test_result["predictions"],
        test_result["probabilities"],
    )

    metrics["Test Loss"] = float(test_result["loss"])
    metrics["Best Validation Accuracy"] = float(best_val_accuracy)
    metrics["Total Training Time (seconds)"] = float(total_training_time)
    metrics["Average Epoch Time (seconds)"] = float(
        total_training_time / max(EPOCHS - start_epoch + 1, 1)
    )

    # --------------------------------------------------------
    # Computational evaluation
    # --------------------------------------------------------
    print("\nMeasuring inference performance...")

    computational_metrics = measure_inference(model, test_loader)
    metrics.update(computational_metrics)

    # --------------------------------------------------------
    # Save metrics and predictions
    # --------------------------------------------------------
    with open(RESULTS_DIR / "evaluation_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    report = classification_report(
        test_result["labels"],
        test_result["predictions"],
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        RESULTS_DIR / "classification_report.csv"
    )

    pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(RESULTS_DIR / "confusion_matrix.csv")

    predictions_df = pd.DataFrame({
        "True Label": [
            CLASS_NAMES[x] for x in test_result["labels"]
        ],
        "Predicted Label": [
            CLASS_NAMES[x] for x in test_result["predictions"]
        ],
    })

    for class_id, class_name in enumerate(CLASS_NAMES):
        predictions_df[f"Probability_{class_name}"] = (
            test_result["probabilities"][:, class_id]
        )

    predictions_df.to_csv(
        RESULTS_DIR / "test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nClassification report:")
    print(
        classification_report(
            test_result["labels"],
            test_result["predictions"],
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(cm)

    print("\nAll outputs saved in:", OUTPUT_DIR.resolve())
