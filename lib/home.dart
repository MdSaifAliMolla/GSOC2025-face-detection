import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tflite/flutter_tflite.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'dart:async';
import 'package:live_emotion_detection/main.dart';
import 'dart:ui';

class Home extends StatefulWidget {
  const Home({Key? key}) : super(key: key);

  @override
  _HomeState createState() => _HomeState();
}

class _HomeState extends State<Home> {
  CameraImage? cameraImage;
  CameraController? cameraController;
  String output = '';
  bool isUsingFrontCamera = true;
  bool _isModelRunning = false;
  Timer? _modelExecutionTimer;
  bool _modelLoaded = false;
  String _errorMessage = '';
  
  // Face detection
  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableContours: false,
      enableClassification: true,
      enableLandmarks: false,
      enableTracking: true,
      performanceMode: FaceDetectorMode.fast,
    ),
  );
  List<Face> _faces = [];
  bool _isFaceDetecting = false;
  
  // Brightness control
  double _currentBrightness = 0.0;
  double _minBrightness = -1.0;
  double _maxBrightness = 1.0;
  bool _isAutoBrightnessEnabled = true;

  @override
  void initState() {
    super.initState();
    loadModel().then((_) {
      loadCamera();
    });
  }

  @override
  void dispose() {
    _modelExecutionTimer?.cancel();
    stopImageStream();
    cameraController?.dispose();
    Tflite.close();
    _faceDetector.close();
    super.dispose();
  }

  Future<void> loadModel() async {
    try {
      await Tflite.loadModel(
        model: "assets/model.tflite",
        labels: "assets/labels.txt",
      );
      _modelLoaded = true;
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to load model';
      });
    }
  }

  void loadCamera() {
    stopImageStream();
    int cameraIndex = isUsingFrontCamera ? 1 : 0;

    cameraController = CameraController(
      cameras![cameraIndex],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    cameraController!.initialize().then((_) async {
      if (!mounted) return;

      await cameraController!.setExposureMode(ExposureMode.auto);
      
      // Get exposure limits
      try {
        _minBrightness = await cameraController!.getMinExposureOffset();
        _maxBrightness = await cameraController!.getMaxExposureOffset();
        _currentBrightness = (_minBrightness + _maxBrightness) / 2;
      } catch (e) {
        print("Error getting exposure limits: $e");
      }
      
      startImageStream();
      setState(() {});
    }).catchError((e) {
      setState(() {
        _errorMessage = 'Camera initialization failed';
      });
    });
  }

  void startImageStream() {
    cameraController!.startImageStream((imageStream) {
      cameraImage = imageStream;
      
      // Run face detection
      detectFaces(imageStream);

      // Run emotion detection with throttling
      if (_modelExecutionTimer == null || !_modelExecutionTimer!.isActive) {
        _modelExecutionTimer = Timer(const Duration(milliseconds: 500), () {
          if (!_isModelRunning && _modelLoaded) {
            runModel();
          }
        });
      }
    });
  }
  
  Future<List<Face>> detectFaces(CameraImage image) async {
  final faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableContours: false,
      enableClassification: true,
      enableLandmarks: false,
      enableTracking: true,
      performanceMode: FaceDetectorMode.fast,
    ),
  );

  try {
    final WriteBuffer allBytes = WriteBuffer();
    for (final Plane plane in image.planes) {
      allBytes.putUint8List(plane.bytes);
    }
    final bytes = allBytes.done().buffer.asUint8List();

    final inputImage = InputImage.fromBytes(
      bytes: bytes,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: InputImageRotation.rotation90deg, 
        format: InputImageFormat.yv12,
        bytesPerRow: image.planes.first.bytesPerRow,
      ),
    );

    final List<Face> faces = await faceDetector.processImage(inputImage);
    return faces;
  } catch (e) {
    print('Face detection failed: $e');
    return [];
  } finally {
    faceDetector.close();
  }
}


  void stopImageStream() {
    if (cameraController != null &&
        cameraController!.value.isInitialized &&
        cameraController!.value.isStreamingImages) {
      cameraController!.stopImageStream();
    }
  }

  void toggleCamera() {
    setState(() {
      isUsingFrontCamera = !isUsingFrontCamera;
      loadCamera();
    });
  }
  
  void toggleAutoBrightness() {
    setState(() {
      _isAutoBrightnessEnabled = !_isAutoBrightnessEnabled;
      if (_isAutoBrightnessEnabled) {
        cameraController?.setExposureMode(ExposureMode.auto);
      } else {
        cameraController?.setExposureMode(ExposureMode.locked);
      }
    });
  }
  
  Future<void> setBrightness(double value) async {
    if (cameraController != null && 
        cameraController!.value.isInitialized) {
      try {
        value = value.clamp(_minBrightness, _maxBrightness);
        
        await cameraController!.setExposureOffset(value);
        
        setState(() {
          _currentBrightness = value;
        });
      } catch (e) {
        print('Error setting exposure: $e');
      }
    }
  }

  Future<void> runModel() async {
    if (cameraImage != null && !_isModelRunning) {
      _isModelRunning = true;
      try {
        var predictions = await Tflite.runModelOnFrame(
          bytesList: cameraImage!.planes.map((plane) => plane.bytes).toList(),
          imageHeight: cameraImage!.height,
          imageWidth: cameraImage!.width,
          imageMean: 127.5,
          imageStd: 127.5,
          rotation: 90,
          numResults: 2,
          threshold: 0.1,
          asynch: true,
        );

        if (predictions != null && predictions.isNotEmpty) {
          setState(() {
            output = predictions.first['label'];
          });
        }
      } catch (e) {
        setState(() {
          _errorMessage = 'TFLite error';
        });
      } finally {
        _isModelRunning = false;
      }
    }
  }
  
  // Get brightness label text
  String getBrightnessLabel() {
    if (_currentBrightness < (_minBrightness * 0.5)) {
      return "Low Light";
    } else if (_currentBrightness > (_maxBrightness * 0.5)) {
      return "Bright";
    } else {
      return "Normal";
    }
  }
  
  // Get brightness indicator color
  Color getBrightnessColor() {
    if (_currentBrightness < (_minBrightness * 0.5)) {
      return Colors.blue;
    } else if (_currentBrightness > (_maxBrightness * 0.5)) {
      return Colors.orange;
    } else {
      return Colors.green;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            const Text('Live Emotion Detection',
            style: TextStyle(
              color: Colors.deepPurpleAccent,
              fontSize: 18,
              fontWeight: FontWeight.w900
            ),),
            IconButton(onPressed: toggleCamera, icon: Icon(Icons.flip_camera_android))
          ],
        ),
      ),
      body: Column(
        children: [
          if (_errorMessage.isNotEmpty)
            Container(
              color: Colors.red.shade100,
              child: Row(
                children: [
                  const Icon(Icons.error_outline, color: Colors.red),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      _errorMessage,
                      style: TextStyle(color: Colors.red.shade900),
                    ),
                  ),
                ],
              ),
            ),
          Stack(
            children: [
              Container(
                height: MediaQuery.of(context).size.height * 0.6,
                width: MediaQuery.of(context).size.width,
                decoration: BoxDecoration(
                  border: Border.all(color: getBrightnessColor(), width: 3),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: cameraController != null &&
                          cameraController!.value.isInitialized
                      ? CameraPreview(cameraController!)
                      : const Center(child: CircularProgressIndicator()),
                ),
              ),
              // Face detection boxes overlay
              if (cameraController != null && cameraController!.value.isInitialized)
                Positioned.fill(
                  child: CustomPaint(
                    painter: FaceDetectionPainter(
                      _faces,
                      cameraController!.value.previewSize!,
                      MediaQuery.of(context).size,
                      isUsingFrontCamera,
                    ),
                  ),
                ),
            ],
          ),
          
          // Brightness indicator
          Container(
            padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 16),
            margin: const EdgeInsets.only(top: 4),
            decoration: BoxDecoration(
              color: getBrightnessColor().withOpacity(0.2),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.brightness_medium, color: getBrightnessColor()),
                const SizedBox(width: 4),
                Text(
                  getBrightnessLabel(),
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: getBrightnessColor(),
                  ),
                ),
              ],
            ),
          ),
          
          // Brightness slider
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Row(
              children: [
                Icon(Icons.brightness_low, size: 16),
                Expanded(
                  child: Slider(
                    value: _currentBrightness,
                    min: _minBrightness,
                    max: _maxBrightness,
                    divisions: 20,
                    activeColor: getBrightnessColor(),
                    label: _currentBrightness.toStringAsFixed(1),
                    onChanged: _isAutoBrightnessEnabled
                        ? null
                        : (value) => setBrightness(value),
                  ),
                ),
                Icon(Icons.brightness_high, size: 16),
              ],
            ),
          ),
          
          // Auto brightness switch
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text("Auto Brightness"),
              Switch(
                value: _isAutoBrightnessEnabled,
                onChanged: (value) => toggleAutoBrightness(),
                activeColor: getBrightnessColor(),
              ),
            ],
          ),
          
          Container(
            padding: const EdgeInsets.all(15),
            decoration: BoxDecoration(
              color: Colors.amberAccent,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(
              output.isEmpty ? "Detecting..." : "Emotion: $output",
              style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
            ),
          )
        ],
      ),
    );
  }
}

// Custom painter for drawing face detection boxes
class FaceDetectionPainter extends CustomPainter {
  final List<Face> faces;
  final Size imageSize;
  final Size widgetSize;
  final bool isUsingFrontCamera;

  FaceDetectionPainter(this.faces, this.imageSize, this.widgetSize, this.isUsingFrontCamera);

  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..color = Colors.yellow
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    for (final Face face in faces) {

      double scaleX = widgetSize.width / (isUsingFrontCamera ? imageSize.height : imageSize.height);
      double scaleY = widgetSize.height / (isUsingFrontCamera ? imageSize.width : imageSize.width);
      
      double left = face.boundingBox.left * scaleX;
      double top = face.boundingBox.top * scaleY;
      double right = face.boundingBox.right * scaleX;
      double bottom = face.boundingBox.bottom * scaleY;
      
      // Flip for front camera
      if (isUsingFrontCamera) {
        left = widgetSize.width - left;
        right = widgetSize.width - right;
        // Swap left and right
        double temp = left;
        left = right;
        right = temp;
      }

      canvas.drawRect(
        Rect.fromLTRB(left, top, right, bottom),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(FaceDetectionPainter oldDelegate) => 
      oldDelegate.faces != faces;
}