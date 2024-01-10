from flask import Flask, render_template, request, jsonify
import numpy as np

import os  # Add this line

app = Flask(__name__)

# Add this line to set the template folder path
app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve_linear_system', methods=['POST'])
def solve_linear_system():
    try:
        data = request.get_json()

        matrix_str = data['matrix']
        vector_str = data['vector']
        method = data['method']

        matrix = np.array([list(map(float, row.split(','))) for row in matrix_str.split(';')])
        vector = np.array(list(map(float, vector_str.split(','))))

        if method == 'Gauss':
            result = np.linalg.solve(matrix, vector)
        elif method == 'Jacobi':
            initial_guess = np.zeros_like(vector)
            result = jacobi_method(matrix, vector, initial_guess)
        elif method == 'Gauss-Seidel':
            initial_guess = np.zeros_like(vector)
            result = gauss_seidel_method(matrix, vector, initial_guess)
        else:
            raise ValueError("Invalid method selected.")

        return jsonify({'result': result.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

def jacobi_method(A, b, initial_guess, tolerance=1e-10, max_iterations=100):
    x = initial_guess.copy()
    D = np.diag(np.diag(A))
    LU = A - D
    for _ in range(max_iterations):
        x_new = np.linalg.inv(D).dot(b - LU.dot(x))
        if np.linalg.norm(x_new - x) < tolerance:
            return x_new
        x = x_new
    raise Exception("La méthode Jacobi n'a pas convergé dans la tolérance et les itérations spécifiées.")

def gauss_seidel_method(A, b, initial_guess, tolerance=1e-10, max_iterations=100):
    x = initial_guess.copy()
    for _ in range(max_iterations):
        for i in range(len(x)):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
        if np.linalg.norm(A.dot(x) - b) < tolerance:
            return x
    raise Exception("La méthode de Gauss-Seidel n’a pas convergé dans les limites de tolérance et d’itérations spécifiées.")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

