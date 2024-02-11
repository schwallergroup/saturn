import numpy as np
import math
from scipy.interpolate import interp1d

from reinvent_scoring.scoring.enums import TransformationTypeEnum
from reinvent_scoring.scoring.enums import TransformationParametersEnum 

class RewardShapingFunction:

    def __init__(self):
        self._transformation_function_registry = self._default_transformation_function_registry()

        # class RewardShapingFunctionParameters:
        # name: str
        # parameters: List[dict]
        # parallel: bool = False

    def _default_transformation_function_registry(self) -> dict:
        enum = TransformationTypeEnum()
        transformation_list = {
            enum.SIGMOID: self.sigmoid_transformation,
            enum.REVERSE_SIGMOID: self.reverse_sigmoid_transformation,
            enum.DOUBLE_SIGMOID: self.double_sigmoid,
            enum.NO_TRANSFORMATION: self.no_transformation,
            enum.RIGHT_STEP: self.right_step,
            enum.LEFT_STEP: self.left_step,
            enum.STEP: self.step,
            enum.CUSTOM_INTERPOLATION: self.custom_interpolation
        }
        return transformation_list

    def get_transformation_function(self, parameters: dict):
        transformation_type = parameters[TransformationParametersEnum.TRANSFORMATION_TYPE]
        transformation_function = self._transformation_function_registry[transformation_type]
        return transformation_function

    def no_transformation(self, predictions: list, parameters: dict) -> np.array:
        return np.array(predictions, dtype=np.float32)

    def right_step(self, predictions, parameters) -> np.array:
        _low = parameters[TransformationParametersEnum.LOW]

        def _right_step_formula(value, low):
            if value >= low:
                return 1
            return 0

        transformed = [_right_step_formula(value, _low) for value in predictions]
        return np.array(transformed, dtype=np.float32)

    def left_step(self, predictions, parameters) -> np.array:
        _low = parameters[TransformationParametersEnum.LOW]

        def _left_step_formula(value, low):
            if value <= low:
                return 1
            return 0

        transformed = [_left_step_formula(value, _low) for value in predictions]
        return np.array(transformed, dtype=np.float32)

    def step(self, predictions, parameters) -> np.array:
        _low = parameters[TransformationParametersEnum.LOW]
        _high = parameters[TransformationParametersEnum.HIGH]

        def _step_formula(value, low, high):
            if low <= value <= high:
                return 1
            return 0

        transformed = [_step_formula(value, _low, _high) for value in predictions]
        return np.array(transformed, dtype=np.float32)

    def sigmoid_transformation(self, predictions: list, parameters: dict) -> np.array:
        _low = parameters[TransformationParametersEnum.LOW]
        _high = parameters[TransformationParametersEnum.HIGH]
        _k = parameters[TransformationParametersEnum.K]

        def _exp(pred_val, low, high, k) -> float:
            return math.pow(10, (10 * k * (pred_val - (low + high) * 0.5) / (low - high)))

        transformed = [1 / (1 + _exp(pred_val, _low, _high, _k)) for pred_val in predictions]
        return np.array(transformed, dtype=np.float32)

    def reverse_sigmoid_transformation(self, predictions: list, parameters: dict) -> np.array:
        _low = parameters[TransformationParametersEnum.LOW]
        _high = parameters[TransformationParametersEnum.HIGH]
        _k = parameters[TransformationParametersEnum.K]

        def _reverse_sigmoid_formula(value, low, high, k) -> float:
            try:
                return 1 / (1 + 10 ** (k * (value - (high + low) / 2) * 10 / (high - low)))
            except:
                return 0

        transformed = [_reverse_sigmoid_formula(pred_val, _low, _high, _k) for pred_val in predictions]
        return np.array(transformed, dtype=np.float32)

    def double_sigmoid(self, predictions: list, parameters: dict) -> np.array:
        _low = parameters[TransformationParametersEnum.LOW]
        _high = parameters[TransformationParametersEnum.HIGH]
        _coef_div = parameters[TransformationParametersEnum.COEF_DIV]
        _coef_si = parameters[TransformationParametersEnum.COEF_SI]
        _coef_se = parameters[TransformationParametersEnum.COEF_SE]

        def _double_sigmoid_formula(value, low, high, coef_div=100., coef_si=150., coef_se=150.):
            try:
                A = 10 ** (coef_se * (value / coef_div))
                B = (10 ** (coef_se * (value / coef_div)) + 10 ** (coef_se * (low / coef_div)))
                C = (10 ** (coef_si * (value / coef_div)) / (
                        10 ** (coef_si * (value / coef_div)) + 10 ** (coef_si * (high / coef_div))))
                return (A / B) - C
            except:
                return 0

        transformed = [_double_sigmoid_formula(pred_val, _low, _high, _coef_div, _coef_si, _coef_se) for pred_val in
                       predictions]
        return np.array(transformed, dtype=np.float32)

    def custom_interpolation(self, predictions: list, parameters: dict) -> np.array:
        """Adapted from the paper:
        'Efficient Multi-Objective Molecular Optimization in a Continuous Latent Space'
        by Robin Winter, Floriane Montanari, Andreas Steffen, Hans Briem, Frank Noé and Djork-Arné Clevert.
        """

        def _transformation_function(interpolation_map, truncate_left=True, truncate_right=True):
            origin = [point['origin'] for point in interpolation_map]
            destination = [point['destination'] for point in interpolation_map]
            assert len(origin) == len(destination)

            if truncate_left:
                origin = [origin[0] - 1] + origin
                destination = [destination[0]] + destination
            if truncate_right:
                origin.append(origin[-1] + 1)
                destination.append(destination[-1])
            return interp1d(origin, destination, fill_value='extrapolate')

        desirability = parameters.get(TransformationParametersEnum.INTERPOLATION_MAP, [{"origin": 0.0, "destination": 0.0},
                                                                         {"origin": 1.0, "destination": 1.0}])
        truncate_left = parameters.get(TransformationParametersEnum.TRUNCATE_LEFT, True)
        truncate_right = parameters.get(TransformationParametersEnum.TRUNCATE_RIGHT, True)

        transformation = _transformation_function(desirability, truncate_left, truncate_right)
        transformed = transformation(predictions)

        return transformed