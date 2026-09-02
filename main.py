import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["relation-probe", "localization", "generalization"],
        default="relation-probe",
    )
    experiment = parser.parse_args().experiment

    if experiment == "localization":
        from src.causal_localization import run
    elif experiment == "generalization":
        from src.position_matched_generalization import run
    else:
        from src.position_matched_relation_probe import run
    run()


if __name__ == "__main__":
    main()
