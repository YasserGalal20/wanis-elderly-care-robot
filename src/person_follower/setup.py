from setuptools import find_packages, setup

package_name = 'person_follower'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        # Runtime deps for final_person_follower.py — installed via pip
        # because rosdep cannot resolve them. The node falls back
        # gracefully when they are missing (candidate-less mode).
        'ultralytics>=8.3.0',   # YOLO11x-seg + built-in BoT-SORT
        'torch>=2.0',           # backs torchreid + ultralytics
        'torchreid>=0.2.5',     # OSNet appearance embeddings
        'scikit-image>=0.20',   # Local Binary Pattern texture signature
    ],
    zip_safe=True,
    maintainer='Yasser Galal',
    maintainer_email='yasser17galal2003@gmail.com',
    description='Identity-locked person following, fall detection and safety guard for the Wanis elderly-care robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "final_person_follower=person_follower.final_person_follower:main",
            "fall_detection=person_follower.fall3:main",
            "safety_guard=person_follower.safety_guard_on_rpi:main",
        ],
    },
)
