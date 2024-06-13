import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Function to validate measurements
def validate_measurements(shape, dimensions, load_type, beam_length):
    if any(d < 0 for d in dimensions):
        return False, "Dimensions cannot be negative."

    if load_type == "Uniformly Distributed Load":
        start, end = dimensions[0], dimensions[1]
        if start > end:
            return False, "Start of load cannot be after end of load."

    elif load_type == "Point Load":
        a = dimensions[0]
        if a > beam_length:
            return False, "Point load position cannot exceed the length of the beam."

    return True, ""

# Function to calculate moment of inertia
def calcMomentOfInertia(shape, dimensions):
    if shape == 'Circle':
        d = dimensions[0]
        I = (np.pi * d**4) / 64
    elif shape == 'Square':
        a = dimensions[0]
        I = (a**4) / 12
    elif shape == 'Rectangle':
        b, h = dimensions
        I = (b * h**3) / 12
    elif shape == 'I-Beam':
        b_f, h_f, b_w, h_w = dimensions
        I_f = (b_f * h_f**3) / 12
        I_w = (b_w * h_w**3) / 12
        I = 2 * I_f + I_w  # Simplified for demonstration
    return I

# Function to calculate deflection and rotation for point load
def calcDeflectionPointLoad(P, L, a, EI, delX):
    X = np.arange(0, L + delX, delX)
    M = np.zeros_like(X)
    for i, x in enumerate(X):
        if x < a:
            M[i] = P * x
        else:
            M[i] = P * a

    return calcDeflection(M, EI, delX, 0, 0, 0), X

# Function to calculate deflection and rotation for uniformly distributed load
def calcDeflectionUDL(w, L, start, end, EI, delX):
    X = np.arange(0, L + delX, delX)
    M = np.zeros_like(X)
    for i, x in enumerate(X):
        if start <= x <= end:
            M[i] = w * (x - start)**2 / 2
        elif x > end:
            M[i] = w * (end - start) * (x - end / 2)
    return calcDeflection(M, EI, delX, 0, 0, 0), X

# Function to calculate deflection and rotation
def calcDeflection(M, EI, delX, theta_0, v_0, supportIndexA):
    theta_im1 = theta_0
    v_im1 = v_0

    Rotation = np.zeros(len(M))
    Rotation[supportIndexA] = theta_im1
    Deflection = np.zeros(len(M))
    Deflection[supportIndexA] = v_im1

    for i, m in enumerate(M[supportIndexA:]):
        ind = i + supportIndexA
        if i > 0:
            M_im1 = M[ind - 1]
            M_i = M[ind]
            M_avg = 0.5 * (M_i + M_im1)

            theta_i = theta_im1 + (M_avg / EI) * delX
            v_i = v_im1 + 0.5 * (theta_i + theta_im1) * delX

            Rotation[ind] = theta_i
            Deflection[ind] = v_i

            theta_im1 = theta_i
            v_im1 = v_i

    return Rotation, Deflection

# Streamlit UI
st.markdown("<h1 style='color:blue;'>Beam Deflection Calculator</h1>", unsafe_allow_html=True)

st.markdown("""
<div style="background-color:lightblue;padding:10px;border-radius:10px;">
This application calculates the deflection and rotation of a beam that is supported by two fixed ends based on its cross-sectional shape, load type, 
and other parameters. Please select the appropriate options and enter the required values to see the results.
</div>
""", unsafe_allow_html=True)

st.image("images/fixed_ends.jpg", caption="Fixed end beam", width=200)

shape = st.selectbox("Select the cross-sectional shape of the beam:", ["Circle", "Square", "Rectangle", "I-Beam"])

if shape == 'Circle':
    d = st.number_input("Enter the diameter (d) in meters:", value=0.1)
    dimensions = [d]
elif shape == 'Square':
    a = st.number_input("Enter the side length (a) in meters:", value=0.1)
    dimensions = [a]
elif shape == 'Rectangle':
    b = st.number_input("Enter the base width (b) in meters:", value=0.1)
    h = st.number_input("Enter the height (h) in meters:", value=0.2)
    dimensions = [b, h]
elif shape == 'I-Beam':
    b_f = st.number_input("Enter the flange width (b_f) in meters:", value=0.1)
    h_f = st.number_input("Enter the flange height (h_f) in meters:", value=0.02)
    b_w = st.number_input("Enter the web width (b_w) in meters:", value=0.02)
    h_w = st.number_input("Enter the web height (h_w) in meters:", value=0.1)
    dimensions = [b_f, h_f, b_w, h_w]

I = calcMomentOfInertia(shape, dimensions)
E = st.number_input("Enter the Young's Modulus (E) in Pascals:", value=210000000000.0)
EI = E * I
st.write(f"Calculated Moment of Inertia (I): {I:.4e} m^4")
st.write(f"Flexural Rigidity (EI): {EI:.4e} N*m^2")

L = st.number_input("Enter the length of the beam (L) in meters:", value=10.0)
delX = st.number_input("Enter the step size (delX) in meters:", value=0.1)
load_type = st.selectbox("Select the load type:", ["Point Load", "Uniformly Distributed Load"])

# Display images based on load type
if load_type == "Point Load":
    st.image("images/Point_load1.png", caption="Point Load", use_column_width=True)
else:
    st.image("images/Uniform_load1.png", caption="Uniformly Distributed Load", use_column_width=True)

P = None
a = None
w = None
start = None
end = None

if load_type == "Point Load":
    P = st.number_input("Enter the magnitude of the point load (P) in Newtons:", value=1000.0)
    a = st.number_input("Enter the distance from the left support to the point load (a) in meters:", value=5.0)
    dimensions = [a]
else:
    w = st.number_input("Enter the magnitude of the uniformly distributed load (w) in N/m:", value=500.0)
    start = st.number_input("Enter the start position of the uniform load (m):", value=2.0)
    end = st.number_input("Enter the end position of the uniform load (m):", value=8.0)
    dimensions = [start, end]

valid, error_message = validate_measurements(shape, dimensions, load_type, L)

if not valid:
    st.error(error_message)
else:
    st.success("Dimensions validated successfully.")
    
    if st.button("Calculate"):
        if load_type == "Point Load":
            X = np.arange(0, L + delX, delX)
            (Rotation, Deflection), X = calcDeflectionPointLoad(P, L, a, EI, delX)
        else:
            X = np.arange(0, L + delX, delX)
            (Rotation, Deflection), X = calcDeflectionUDL(w, L, start, end, EI, delX)

        # Create a dataframe to display the results
        results = pd.DataFrame({
            'Position (m)': X,
            'Rotation (radians)': Rotation,
            'Deflection (m)': Deflection
        })
        
        st.write("### Results:")
        st.dataframe(results)

        # Plot the results using Plotly for better interactivity
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=X, y=Deflection, mode='lines', name='Deflection (m)', line=dict(color='firebrick', width=4)))
        fig.add_trace(go.Scatter(x=X, y=Rotation, mode='lines', name='Rotation (radians)', line=dict(color='royalblue', width=4)))

        fig.update_layout(
            title="Beam Deflection and Rotation",
            xaxis_title="Position (m)",
            yaxis_title="Value",
            legend_title="Legend",
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="RebeccaPurple"
            )
        )

        st.plotly_chart(fig)

        st.markdown("""
        <div style="background-color:lightgreen;padding:10px;border-radius:10px;">
        <h4>Note:</h4>
        <ul>
        <li>The position, rotation, and deflection values are calculated and displayed in the table above.</li>
        <li>The plot shows the rotation (blue line) and deflection (red line) of the beam along its length.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
       


   
