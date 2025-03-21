// import 'package:camera/camera.dart';
// import 'package:flutter/material.dart';
// import 'package:live_emotion_detection/main.dart';
// import 'package:flutter_tflite/flutter_tflite.dart';

// class Home extends StatefulWidget {
//   const Home({Key? key}) : super(key: key);

//   @override
//   _HomeState createState() => _HomeState();
// }

// class _HomeState extends State<Home> {
//   CameraImage? cameraImage;
//   CameraController? cameraController;
//   String output = '';

//   @override
//   void initState() {
//     super.initState();
//     loadCamera();
//     loadmodel();
//   }

//   loadCamera() {
//     cameraController = CameraController(cameras![1], ResolutionPreset.high);
//     cameraController!.initialize().then((value) {
//       if (!mounted) {
//         return;
//       } else {
//         setState(() {
//           cameraController!.startImageStream((imageStream) {
//             cameraImage = imageStream;
//             runModel();
//           });
//         });
//       }
//     });
//   }

//   runModel() async {
//     if (cameraImage != null) {
//       var predictions = await Tflite.runModelOnFrame(
//           bytesList: cameraImage!.planes.map((plane) {
//             return plane.bytes;
//           }).toList(),
//           imageHeight: cameraImage!.height,
//           imageWidth: cameraImage!.width,
//           imageMean: 127.5,
//           imageStd: 127.5,
//           rotation: 90,
//           numResults: 2,
//           threshold: 0.1,
//           asynch: true);
//       predictions!.forEach((element) {
//         setState(() {
//           output = element['label'];
//         });
//       });
//     }
//     print("[INFO]===============model running");
//   }

//   loadmodel() async {
//     await Tflite.loadModel(
//         model: "assets/emo.tflite", labels: "assets/labels.txt");
//         print('[INFO]============model loaded');
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(title: Text('Live From Camera'),
//         centerTitle: true,
//       ),
//       body: Column(children: [
//         Padding(
//           padding: EdgeInsets.all(20),
//           child: Container(
//             height: MediaQuery.of(context).size.height * 0.7,
//             width: MediaQuery.of(context).size.width,
//             child: !cameraController!.value.isInitialized
//                 ? Container()
//                 : AspectRatio(
//               aspectRatio: cameraController!.value.aspectRatio,
//               child: CameraPreview(cameraController!),
//             ),
//           ),
//         ),
//         Text(
//           output,
//           style: TextStyle(fontWeight: FontWeight.bold, fontSize: 20),
//         )
//       ]),
//     );
//   }
// }