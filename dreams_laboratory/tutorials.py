from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import random
import numpy as np

@dataclass
class WidgetParameters:
    """Base class for widget parameters"""
    pass

@dataclass
class StereoParameters(WidgetParameters):
    baseline: float  # meters
    focal_length: float  # mm
    sensor_width: float  # mm
    point_depth: float  # meters
    noise_level: float = 0.0
    num_points: int = 10

@dataclass
class RansacParameters(WidgetParameters):
    num_points: int
    outlier_ratio: float
    noise_std: float
    model_params: Dict[str, float]

@dataclass
class ValidationRule:
    type: str  # 'numeric', 'categorical', 'array'
    tolerance: float = 0.1  # For numeric validation
    options: List[str] = None  # For categorical validation
    array_tolerance: float = 0.1  # For array validation

class InteractiveComponent:
    """Base class for all interactive components (tutorials and quizzes)"""
    def __init__(
        self,
        component_id: str,
        title: str,
        description: str,
        widget_type: str,
        parameters: WidgetParameters,
        validation_rules: ValidationRule
    ):
        self.id = component_id
        self.title = title
        self.description = description
        self.widget_type = widget_type
        self.parameters = parameters
        self.validation_rules = validation_rules

    def get_context(self, mode: str = 'tutorial') -> Dict[str, Any]:
        """Get the context for rendering the component"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'widget_type': self.widget_type,
            'parameters': self.parameters.__dict__,
            'mode': mode,
            'show_hints': mode == 'tutorial'
        }

class StereoVisionComponent(InteractiveComponent):
    """Stereo vision tutorial/quiz component"""
    def __init__(
        self,
        component_id: str,
        difficulty: str = 'medium',
        mode: str = 'tutorial'
    ):
        # Base parameters
        base_params = StereoParameters(
            baseline=0.2,  # 20cm baseline
            focal_length=50.0,  # 50mm lens
            sensor_width=36.0,  # Full frame sensor
            point_depth=2.0,  # 2 meters depth
            noise_level=0.0,
            num_points=10
        )

        # Adjust parameters based on difficulty
        if difficulty == 'easy':
            params = StereoParameters(
                **{**base_params.__dict__, 
                   'point_depth': 1.5,
                   'noise_level': 0.0,
                   'num_points': 5}
            )
        elif difficulty == 'medium':
            params = StereoParameters(
                **{**base_params.__dict__,
                   'point_depth': 2.5,
                   'noise_level': 0.02,
                   'num_points': 10}
            )
        else:  # hard
            params = StereoParameters(
                **{**base_params.__dict__,
                   'point_depth': 3.5,
                   'noise_level': 0.05,
                   'num_points': 15}
            )

        # Validation rules
        validation = ValidationRule(
            type='numeric',
            tolerance=0.1 if difficulty == 'easy' else 0.05
        )

        super().__init__(
            component_id=component_id,
            title='Stereo Vision Challenge',
            description=self._get_description(mode, difficulty),
            widget_type='stereo-buddy',
            parameters=params,
            validation_rules=validation
        )

    def _get_description(self, mode: str, difficulty: str) -> str:
        """Get appropriate description based on mode and difficulty"""
        if mode == 'tutorial':
            return """
            Learn about stereo vision by estimating depths from stereo image pairs.
            Use the disparity between corresponding points to calculate depth.
            """
        else:  # quiz mode
            if difficulty == 'easy':
                return "Estimate the depth of the marked point in the stereo images."
            elif difficulty == 'medium':
                return "Estimate depths of multiple points with some noise present."
            else:
                return "Estimate depths with significant noise and outliers present."

    def generate_ground_truth(self) -> Dict[str, Any]:
        """Generate ground truth data for validation"""
        params = self.parameters
        # Calculate disparity
        disparity = params.baseline * params.focal_length / params.point_depth
        # Add noise if specified
        if params.noise_level > 0:
            disparity += np.random.normal(0, params.noise_level, 1)[0]
        return {
            'disparity': disparity,
            'depth': params.point_depth
        }

class RansacComponent(InteractiveComponent):
    """RANSAC tutorial/quiz component"""
    def __init__(
        self,
        component_id: str,
        difficulty: str = 'medium',
        mode: str = 'tutorial'
    ):
        # Base parameters for line fitting
        base_params = RansacParameters(
            num_points=50,
            outlier_ratio=0.2,
            noise_std=0.1,
            model_params={'slope': 1.0, 'intercept': 0.0}
        )

        # Adjust parameters based on difficulty
        if difficulty == 'easy':
            params = RansacParameters(
                **{**base_params.__dict__,
                   'num_points': 30,
                   'outlier_ratio': 0.1,
                   'noise_std': 0.05}
            )
        elif difficulty == 'medium':
            params = base_params
        else:  # hard
            params = RansacParameters(
                **{**base_params.__dict__,
                   'num_points': 100,
                   'outlier_ratio': 0.4,
                   'noise_std': 0.2}
            )

        # Validation rules
        validation = ValidationRule(
            type='array',
            array_tolerance=0.1 if difficulty == 'easy' else 0.05
        )

        super().__init__(
            component_id=component_id,
            title='Random Sample Consensus (RANSAC)',
            description=self._get_description(mode, difficulty),
            widget_type='ransac-buddy',
            parameters=params,
            validation_rules=validation
        )

    def _get_description(self, mode: str, difficulty: str) -> str:
        """Get appropriate description based on mode and difficulty"""
        if mode == 'tutorial':
            return """
            Learn about RANSAC by fitting models to noisy data with outliers.
            Experiment with different thresholds and iteration counts.
            """
        else:  # quiz mode
            if difficulty == 'easy':
                return "Fit a line to the data and identify obvious outliers."
            elif difficulty == 'medium':
                return "Determine the optimal threshold for outlier rejection."
            else:
                return "Handle a challenging dataset with many outliers."

    def generate_data(self) -> Dict[str, np.ndarray]:
        """Generate synthetic data for the RANSAC exercise"""
        params = self.parameters
        num_inliers = int(params.num_points * (1 - params.outlier_ratio))
        num_outliers = params.num_points - num_inliers

        # Generate inlier points
        x_inliers = np.random.uniform(-5, 5, num_inliers)
        y_inliers = (params.model_params['slope'] * x_inliers + 
                    params.model_params['intercept'] + 
                    np.random.normal(0, params.noise_std, num_inliers))

        # Generate outlier points
        x_outliers = np.random.uniform(-5, 5, num_outliers)
        y_outliers = np.random.uniform(-10, 10, num_outliers)

        # Combine and shuffle
        x = np.concatenate([x_inliers, x_outliers])
        y = np.concatenate([y_inliers, y_outliers])
        shuffle_idx = np.random.permutation(len(x))

        return {
            'x': x[shuffle_idx],
            'y': y[shuffle_idx],
            'true_inliers': shuffle_idx < num_inliers
        }

class TutorialManager:
    """Manages the creation and validation of tutorials and quizzes"""
    def __init__(self):
        self.components = {
            'stereo': StereoVisionComponent,
            'ransac': RansacComponent
        }

    def create_tutorial(
        self,
        component_type: str,
        component_id: str,
        difficulty: str = 'medium'
    ) -> InteractiveComponent:
        """Create a tutorial component"""
        if component_type not in self.components:
            raise ValueError(f"Unknown component type: {component_type}")
        
        component_class = self.components[component_type]
        return component_class(component_id, difficulty, mode='tutorial')

    def create_quiz(
        self,
        component_type: str,
        component_id: str,
        difficulty: str = 'medium'
    ) -> InteractiveComponent:
        """Create a quiz component"""
        if component_type not in self.components:
            raise ValueError(f"Unknown component type: {component_type}")
        
        component_class = self.components[component_type]
        return component_class(component_id, difficulty, mode='quiz')

    def validate_answer(
        self,
        component: InteractiveComponent,
        student_answer: Any,
        correct_answer: Any
    ) -> bool:
        """Validate a student's answer against the correct answer"""
        rule = component.validation_rules

        if rule.type == 'numeric':
            try:
                student_val = float(student_answer)
                correct_val = float(correct_answer)
                return abs(student_val - correct_val) <= rule.tolerance
            except (ValueError, TypeError):
                return False

        elif rule.type == 'categorical':
            return student_answer in rule.options and student_answer == correct_answer

        elif rule.type == 'array':
            try:
                student_arr = np.array(student_answer)
                correct_arr = np.array(correct_answer)
                return np.all(np.abs(student_arr - correct_arr) <= rule.array_tolerance)
            except (ValueError, TypeError):
                return False

        return False

# Usage example:
"""
# Create a tutorial manager
manager = TutorialManager()

# Create a tutorial component
tutorial = manager.create_tutorial('stereo', 'stereo_tutorial_1', difficulty='easy')

# Create a quiz component
quiz = manager.create_quiz('stereo', 'stereo_quiz_1', difficulty='hard')

# Get context for rendering
tutorial_context = tutorial.get_context()
quiz_context = quiz.get_context()

# Validate answers
is_correct = manager.validate_answer(quiz, student_answer=2.5, correct_answer=2.3)
""" 