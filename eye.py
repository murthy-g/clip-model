from bokeh.plotting import figure, output_file, show
from bokeh.models import LinearGradient

# Create a new Bokeh plot
output_file("realistic_eye.html")
p = figure(width=400, height=400)

# Define the iris and pupil
iris = p.ellipse(200, 200, width=80, height=120, color="blue", line_color="black")
pupil = p.ellipse(200, 200, width=40, height=40, color="black")

# Define a radial gradient for the iris
iris_gradient = LinearGradient(x0=200, y0=200, x1=200, y1=200, colors=["blue", "lightblue"])
iris.fill_color = {"field": "y", "transform": iris_gradient}

# Add some highlights
highlight1 = p.circle(220, 220, size=10, color="white")
highlight2 = p.circle(180, 220, size=10, color="white")

# Customize the plot appearance
p.axis.visible = False
p.grid.visible = False

# Show the plot
show(p)
