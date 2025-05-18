"""
Quiz Views Module - SES598 Retrospective Quiz functionality
"""
from django.shortcuts import render
import json
import uuid
import logging
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.cache import cache
from dreams_laboratory.models import QuizSubmission  # Import the QuizSubmission model

# Set up logging
logger = logging.getLogger(__name__)

@csrf_exempt
def ses598_2025_retrospective(request):
    """
    View for the SES598 2025 Retrospective Quiz
    """
    # Generate unique session ID if not already present
    session_id = request.session.get('quiz_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['quiz_session_id'] = session_id

    # Generate timestamp for the form
    current_timestamp = timezone.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Initialize context
    context = {
        'session_id': session_id,
        'show_results': False,
        'email': request.session.get('quiz_email', ''),
        'timestamp': current_timestamp
    }

    if request.method == 'POST':
        # Save email for future quiz sessions
        email = request.POST.get('email', 'Anonymous')
        request.session['quiz_email'] = email
        
        logger.info(f"Processing retrospective quiz submission for {email}")
        
        # Process answers
        answers = {}
        for i in range(1, 26):  # We now have 25 questions
            q_key = f'q{i}'
            answers[q_key] = request.POST.get(q_key)
        
        logger.debug(f"Submitted answers: {answers}")
        
        # Get correct answers from the function
        correct_answers = json.loads(get_correct_answers())
        
        logger.debug(f"Correct answers: {correct_answers}")
        
        # Calculate score
        score = 0
        feedback = {}
        
        for q_num, answer in answers.items():
            correct = correct_answers.get(q_num)
            is_correct = (answer == correct)
            
            if is_correct:
                score += 4  # 4% per correct answer (25 questions total)
            
            logger.debug(f"Question {q_num}: submitted={answer}, correct={correct}, is_correct={is_correct}")
                
            # Generate feedback for each question
            feedback[q_num] = {
                'is_correct': is_correct,
                'student_answer': answer,
                'correct_answer': correct,
                'explanation': get_explanation(q_num, is_correct)
            }
        
        logger.info(f"Quiz score for {email}: {score}%")
        
        # Update context with results
        context.update({
            'show_results': True,
            'score': score,
            'feedback': feedback,
            'email': email
        })
        
        # Get the timestamp from the form submission or use current time
        submission_timestamp = request.POST.get('timestamp', current_timestamp)
        
        # Store quiz results in cache
        cache_key = f"2025_retro_quiz_{session_id}"
        cache.set(cache_key, {
            'email': email,
            'score': score,
            'answers': answers,
            'timestamp': submission_timestamp
        }, timeout=60*60*24*30)  # Store for 30 days
        
        # Save quiz results to database
        try:
            # Since our QuizSubmission model only has fields for 15 questions,
            # but our quiz has 25 questions, we'll create two submission records:
            # one for questions 1-15 and another for questions 16-25
            
            # Create first submission for questions 1-15
            quiz_submission_part1 = QuizSubmission(
                quiz_id='SES598_2025_RETRO_P1',  # Part 1 of the retrospective quiz
                session_id=session_id,
                email=email,
                total_score=score,  # Store total score in both records for consistency
                # Categorize scores by domain (simplified for retrospective quiz)
                cv_score=score,  # Using total score for CV since most questions are CV-related
                slam_score=0.0,
                estimation_score=0.0,
                sensing_score=0.0,
                motion_score=0.0,
                neural_score=0.0,
                # Store questions 1-15
                q1=answers.get('q1', ''),
                q2=answers.get('q2', ''),
                q3=answers.get('q3', ''),
                q4=answers.get('q4', ''),
                q5=answers.get('q5', ''),
                q6=answers.get('q6', ''),
                q7=answers.get('q7', ''),
                q8=answers.get('q8', ''),
                q9=answers.get('q9', ''),
                q10=answers.get('q10', ''),
                q11=answers.get('q11', ''),
                q12=answers.get('q12', ''),
                q13=answers.get('q13', ''),
                q14=answers.get('q14', ''),
                q15=answers.get('q15', '')
            )
            quiz_submission_part1.save()
            
            # Create second submission for questions 16-25 (stored in q1-q10 fields)
            quiz_submission_part2 = QuizSubmission(
                quiz_id='SES598_2025_RETRO_P2',  # Part 2 of the retrospective quiz
                session_id=session_id,
                email=email,
                total_score=score,  # Store total score in both records
                cv_score=score,  # Using total score for CV since most questions are CV-related
                slam_score=0.0,
                estimation_score=0.0,
                sensing_score=0.0,
                motion_score=0.0,
                neural_score=0.0,
                # Store questions 16-25 in fields q1-q10
                q1=answers.get('q16', ''),
                q2=answers.get('q17', ''),
                q3=answers.get('q18', ''),
                q4=answers.get('q19', ''),
                q5=answers.get('q20', ''),
                q6=answers.get('q21', ''),
                q7=answers.get('q22', ''),
                q8=answers.get('q23', ''),
                q9=answers.get('q24', ''),
                q10=answers.get('q25', '')
                # q11-q15 will be NULL
            )
            quiz_submission_part2.save()
            
            logger.info(f"Quiz submissions saved to database for {email}")
        except Exception as e:
            logger.error(f"Error saving quiz submissions to database: {str(e)}")
        
    return render(request, 'ses598_2025_retrospective.html', context)

def get_correct_answers():
    """Return correct answers for the quiz in JSON format"""
    return '''{
        "q1": "B",  
        "q2": "C",  
        "q3": "B",  
        "q4": "C",  
        "q5": "A",
        "q6": "D",  
        "q7": "B",  
        "q8": "B",  
        "q9": "C",  
        "q10": "B",
        "q11": "A",  
        "q12": "B",  
        "q13": "B",  
        "q14": "B",  
        "q15": "B",
        "q16": "B",  
        "q17": "A",  
        "q18": "B",  
        "q19": "B",  
        "q20": "B",
        "q21": "B",
        "q22": "B",
        "q23": "C",
        "q24": "C",
        "q25": "D"
    }'''

def get_explanation(q_num, is_correct):
    """Return explanation for each question based on correctness"""
    explanations = {
        'q1': {
            True: "Correct! PAC loop stands for Perception-Action-Communication loop, particularly in the context of robotic swarms. Perception: Robots sense their environment, gather information using sensors (vision, lidar, etc.). Action: Robots execute actions based on sensed data, like navigation, manipulation, or other physical interactions. Communication: Robots share information among themselves, coordinate their actions, and collaborate towards shared goals. Together, these three processes form a continuous feedback loop enabling collective intelligence and autonomous coordination in robotic swarms.",
            False: "Incorrect. The PAC loop (Perception-Action-Communication) describes how robots in a swarm continually sense their environment (perception), perform tasks based on sensed information (action), and share information with other robots (communication), thereby functioning collectively and adaptively. It's a fundamental concept for understanding swarm robotics and distributed systems."
        },
        'q2': {
            True: "Correct! A closed-loop feedback control system continuously measures the output, compares it to the desired state, and adjusts based on the error. Option B describes Feedforward control.",
            False: "Incorrect. A closed-loop feedback control system continuously senses output error and adjusts actions accordingly. Option B describes Feedforward control."
        },
        'q3': {
            True: "Correct! Gaussian Process Regression is a powerful method for estimating uncertainty and making predictions for continuous data.",
            False: "Incorrect. Gaussian Process Regression (GPR) is primarily used for estimating uncertainty and making predictions for continuous data."
        },
        'q4': {
            True: "Correct! Mutual information guides sensor placement to maximize information gain and reduce uncertainty in model estimates.",
            False: "Incorrect. The primary advantage of using mutual information for sensor placement is reducing uncertainty about model estimates."
        },
        'q5': {
            True: "Correct! Active learning focuses on minimizing variance and exploring to learn a model, while multi-armed bandits focus on maximizing expected rewards.",
            False: "Incorrect. Active learning aims to minimize variance in the learned model, while multi-armed bandits aim to maximize expected rewards."
        },
        'q6': {
            True: "Correct! The Kalman Filter is provably optimal for systems with linear dynamics and Gaussian noise.",
            False: "Incorrect. For linear systems with Gaussian noise, the standard Kalman Filter is the optimal state estimator."
        },
        'q7': {
            True: "Correct! Observability refers to the ability to determine all state variables from the system's outputs (sensor measurements).",
            False: "Incorrect. Observability describes whether all state variables can be determined from the system's outputs (sensor measurements)."
        },
        'q8': {
            True: "Correct! In LQR, the Q matrix penalizes state deviations while the R matrix penalizes control input usage.",
            False: "Incorrect. In LQR, Q and R matrices represent the relative weights given to state deviation and control input usage in the cost function."
        },
        'q9': {
            True: "Correct! DUSt3R (Dense Uncalibrated Structure from motion for 3D Reconstruction) directly reconstructs 3D point maps without requiring camera calibration.",
            False: "Incorrect. DUSt3R (Dense Uncalibrated Structure from motion for 3D Reconstruction) directly reconstructs dense 3D point maps without requiring camera calibration."
        },
        'q10': {
            True: "Correct! NeRF enables high-quality 3D scene reconstruction and novel view synthesis from 2D images.",
            False: "Incorrect. Neural Radiance Fields (NeRF) are particularly useful for image-based 3D scene reconstruction and novel view synthesis."
        },
        'q11': {
            True: "Correct! Physics-based terrain simulations must balance accuracy with computational cost.",
            False: "Incorrect. A key trade-off in physics-based terrain simulation is between accuracy and computational expense."
        },
        'q12': {
            True: "Correct! The Fiedler value (second smallest eigenvalue of the Laplacian matrix) represents algebraic connectivity of a graph.",
            False: "Incorrect. The algebraic connectivity of a graph is known as the Fiedler value, which is the second smallest eigenvalue of the Laplacian matrix."
        },
        'q13': {
            True: "Correct! Model Predictive Control (MPC) optimizes future control actions by predicting system behavior over a finite horizon.",
            False: "Incorrect. MPC stands for Model Predictive Control, which optimizes control by predicting system behavior over a finite time horizon."
        },
        'q14': {
            True: "Correct! Importance sampling improves particle filter efficiency by sampling from a proposal distribution rather than the prior.",
            False: "Incorrect. Importance sampling is used in Particle Filtering to efficiently sample from a proposal distribution rather than the true distribution."
        },
        'q15': {
            True: "Correct! Digital Twins are virtual replicas of physical systems that can be used for monitoring, simulation, and optimization in real-time.",
            False: "Incorrect. Digital Twins are accurate real-time simulations that replicate physical systems, used for monitoring, predicting, and optimization."
        },
        'q16': {
            True: "Correct! Limited rover autonomy is a major challenge for long-duration lunar missions due to communication delays and limited human intervention.",
            False: "Incorrect. A core challenge in long-duration robotic lunar missions is limited rover autonomy, which is needed due to communication delays and limited human intervention."
        },
        'q17': {
            True: "Correct! Differential flatness allows expressing states and inputs explicitly through outputs and their derivatives, simplifying trajectory planning.",
            False: "Incorrect. The key benefit of differential flatness is that it allows expressing states and inputs explicitly through outputs and their derivatives."
        },
        'q18': {
            True: "Correct! LQG controllers are designed to handle Gaussian noise in both actuators and sensors.",
            False: "Incorrect. Linear Quadratic Gaussian (LQG) controllers explicitly account for Gaussian noise in both actuators and sensors."
        },
        'q19': {
            True: "Correct! Chrono Camera Models simulate realistic imaging effects including lens distortion and sensor noise within the Chrono simulation environment.",
            False: "Incorrect. Chrono Camera Models framework simulates effects like lens distortion and sensor noise to generate realistic imagery within the Chrono simulation environment."
        },
        'q20': {
            True: "Correct! The exploration-exploitation trade-off balances gathering new information against leveraging known information for optimal results.",
            False: "Incorrect. The exploration-exploitation trade-off refers to balancing between gathering new information (exploration) and leveraging known information (exploitation)."
        },
        'q21': {
            True: "Correct! Epipolar geometry is a key concept in multi-view geometry that concerns the relationship between two camera views and how to estimate the position and orientation of one camera relative to another.",
            False: "Incorrect. Epipolar geometry is primarily concerned with estimating the pose of one camera relative to another camera, fundamental for stereo vision and structure from motion."
        },
        'q22': {
            True: "Correct! DUSt3R's key innovation is unifying depth estimation, pose estimation, and 3D reconstruction into a single process, enabling reconstruction without explicit camera calibration.",
            False: "Incorrect. DUSt3R unifies depth estimation, pose estimation, and 3D reconstruction into one process, which is its distinctive feature in 3D vision."
        },
        'q23': {
            True: "Correct! NeRF optimizes a neural network to predict the color and density of any point in 3D space, allowing for the synthesis of novel views from any perspective.",
            False: "Incorrect. Neural Radiance Fields (NeRF) optimizes a neural network to predict scene radiance from any viewpoint, enabling high-quality novel view synthesis."
        },
        'q24': {
            True: "Correct! Gaussian Splatting achieves real-time rendering capabilities by representing scenes as 3D Gaussian primitives (blobs) that can be efficiently rendered using specialized algorithms.",
            False: "Incorrect. Gaussian Splatting is primarily characterized by its real-time rendering capabilities using 3D Gaussian blobs, offering a speed advantage over traditional NeRF."
        },
        'q25': {
            True: "Correct! COLMAP is an industry-standard Structure-from-Motion (SfM) tool widely used for precise metric reconstructions in cultural heritage, archaeology, and other applications requiring high accuracy.",
            False: "Incorrect. COLMAP is widely used for precise metric reconstructions like cultural heritage documentation, offering high accuracy in calibrated conditions."
        }
    }
    
    return explanations.get(q_num, {}).get(is_correct, "No explanation available.") 