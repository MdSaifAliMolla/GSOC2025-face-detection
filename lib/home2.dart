import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_tflite/flutter_tflite.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class ImageEmotionDetector extends StatefulWidget {
  const ImageEmotionDetector({Key? key}) : super(key: key);

  @override
  State<ImageEmotionDetector> createState() => _ImageEmotionDetectorState();
}

class _ImageEmotionDetectorState extends State<ImageEmotionDetector> {
  File? _image;
  String _output = '';
  bool _isLoading = false;
  Face? _detectedFace;

  final picker = ImagePicker();

  // Face detection
  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableContours: false,
      enableClassification: true,
      enableLandmarks: false,
      enableTracking: false,
      performanceMode: FaceDetectorMode.accurate,
    ),
  );

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  @override
  void dispose() {
    Tflite.close();
    _faceDetector.close();
    super.dispose();
  }

  Future<void> loadModel() async {
    await Tflite.loadModel(
      model: "assets/emotion_model.tflite",
      labels: "assets/l.txt",
    );
  }

  Future<void> pickImage(ImageSource source) async {
    final pickedFile = await picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _output = '';
        _detectedFace = null;
      });

      await detectFaceAndRunModel(File(pickedFile.path));
    }
  }

  Future<void> detectFaceAndRunModel(File imageFile) async {
    setState(() {
      _isLoading = true;
    });

    final inputImage = InputImage.fromFile(imageFile);
    final List<Face> faces = await _faceDetector.processImage(inputImage);

    if (faces.isEmpty) {
      setState(() {
        _output = "No face detected!";
        _isLoading = false;
      });
      return;
    }

    setState(() {
      _detectedFace = faces.first;
    });

    await runModelOnImage(imageFile);
  }

  Future<void> runModelOnImage(File imageFile) async {
    var predictions = await Tflite.runModelOnImage(
      path: imageFile.path,
      imageMean: 127.5,
      imageStd: 127.5,
      numResults: 2,
      threshold: 0.1,
      asynch: true,
    );

    if (predictions != null && predictions.isNotEmpty) {
      setState(() {
        _output = predictions.first['label'];
      });
    } else {
      setState(() {
        _output = "No emotion detected";
      });
    }

    setState(() {
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Image Emotion Detection',
          style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.w900,
              color: Colors.deepPurpleAccent),
        ),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          if (_image != null)
            Container(
              padding: const EdgeInsets.all(8),
              child: Stack(
                children: [
                  Image.file(_image!),
                  if (_detectedFace != null)
                    Positioned.fill(
                      child: CustomPaint(
                        painter: FaceBoxPainter(
                          face: _detectedFace!,
                          imageSize: _image!,
                        ),
                      ),
                    ),
                ],
              ),
            ),
          const SizedBox(height: 20),
          _isLoading
              ? const CircularProgressIndicator()
              : Text(
                  _output.isEmpty ? 'Pick an image' : "Emotion: $_output",
                  style: const TextStyle(
                      fontSize: 20, fontWeight: FontWeight.bold),
                ),
          const SizedBox(height: 30),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              ElevatedButton.icon(
                onPressed: () => pickImage(ImageSource.gallery),
                icon: const Icon(Icons.photo),
                label: const Text("Gallery"),
              ),
              ElevatedButton.icon(
                onPressed: () => pickImage(ImageSource.camera),
                icon: const Icon(Icons.camera_alt),
                label: const Text("Camera"),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class FaceBoxPainter extends CustomPainter {
  final Face face;
  final File imageSize;

  FaceBoxPainter({required this.face, required this.imageSize});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.redAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    // Get the actual image dimensions
    final image = FileImage(imageSize);
    image.resolve(const ImageConfiguration()).addListener(
      ImageStreamListener((ImageInfo info, _) {
        final double scaleX = size.width / info.image.width;
        final double scaleY = size.height / info.image.height;

        final rect = Rect.fromLTRB(
          face.boundingBox.left * scaleX,
          face.boundingBox.top * scaleY,
          face.boundingBox.right * scaleX,
          face.boundingBox.bottom * scaleY,
        );

        canvas.drawRect(rect, paint);
      }),
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
