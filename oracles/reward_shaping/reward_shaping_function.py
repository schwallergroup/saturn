"""
Adapted from https://github.com/MolecularAI/reinvent-scoring.
Implements the following reward shaping functions:
    1. No transformation
    2. Step
    3. Sigmoid
    4. Reverse Sigmoid
    5. Double Sigmoid
"""
import warnings
import numpy as np
import math

from oracles.reward_shaping.function_parameters import RewardShapingFunctionParameters

class RewardShapingFunction:

    def __init__(self, parameters: RewardShapingFunctionParameters):
        self.transformation_function = parameters.transformation_function
        self.parameters = parameters.parameters

        assert self.transformation_function in [
            "no_transformation", 
            "step", 
            "sigmoid", 
            "reverse_sigmoid", 
            "double_sigmoid"
        ], f"{self.transformation_function} reward shaping function is not implemented."

    def __call__(self, raw_property_values: np.ndarray[float]) -> np.ndarray[float]:
        """
        Takes as input the raw property values based on the OracleComponent and applies reward shaping.
        """
        try:
            if self.transformation_function == "no_transformation":
                return raw_property_values
            elif self.transformation_function == "step":
                return self.step_transformation(raw_property_values, **self.parameters)
            elif self.transformation_function == "sigmoid":
                return self.sigmoid_transformation(raw_property_values, **self.parameters)
            elif self.transformation_function == "reverse_sigmoid":
                return self.reverse_sigmoid_transformation(raw_property_values, **self.parameters)
            elif self.transformation_function == "double_sigmoid":
                return self.double_sigmoid_transformation(raw_property_values, **self.parameters)
        except Exception:
            # In case not all required parameters are specified
            raise ValueError(f"Oracle: {self.oracle_name} is using {self.transformation_function} reward shaping function but not all parameters have been specified.")


    def step_transformation(
        self, 
        raw_property_values: np.ndarray[float],
        low: float,
        high: float
    ) -> np.ndarray[float]:

        def _step_formula(value, low, high) -> float:
            if low <= value <= high:
                return 1.0
            return 0.0

        transformed = [_step_formula(val, low, high) for val in raw_property_values]
        return np.array(transformed, dtype=np.float32)

    def sigmoid_transformation(
        self, 
        raw_property_values: np.ndarray[float], 
        low: float,
        high: float,
        k: float
    ) -> np.ndarray[float]:

        def _sigmoid(value, low, high, k) -> float:
            try:
                return math.pow(10, (10 * k * (value - (low + high) * 0.5) / (low - high)))
            except Exception:
                return 0.0

        transformed = [1 / (1 + _sigmoid(val, low, high, k)) for val in raw_property_values]
        return np.array(transformed, dtype=np.float32)

    def reverse_sigmoid_transformation(
        self, 
        raw_property_values: np.ndarray[float], 
        low: float,
        high: float,
        k: float
    ) -> np.ndarray[float]:

        def _reverse_sigmoid_formula(value, low, high, k) -> float:
            try:
                return 1 / (1 + 10 ** (k * (value - (high + low) / 2) * 10 / (high - low)))
            except Exception:
                return 0.0

        transformed = [_reverse_sigmoid_formula(val, low, high, k) for val in raw_property_values]
        return np.array(transformed, dtype=np.float32)

    def double_sigmoid_transformation(
        self, 
        raw_property_values: np.ndarray[float], 
        low: float,
        high: float,
        coef_div: float,
        coef_si: float,
        coef_se: float
    ) -> np.ndarray[float]:

        def _double_sigmoid_formula(value, low, high, coef_div, coef_si, coef_se):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    A = 10 ** (coef_se * (value / coef_div))
                    B = (10 ** (coef_se * (value / coef_div)) + 10 ** (coef_se * (low / coef_div)))
                    C = (10 ** (coef_si * (value / coef_div)) / (
                            10 ** (coef_si * (value / coef_div)) + 10 ** (coef_si * (high / coef_div))))
                    return (A / B) - C
            except RuntimeWarning:
                # In case of numerical overflow
                return 0.0

        transformed = [_double_sigmoid_formula(val, low, high, coef_div, coef_si, coef_se) for val in raw_property_values]
        return np.array(transformed, dtype=np.float32)
