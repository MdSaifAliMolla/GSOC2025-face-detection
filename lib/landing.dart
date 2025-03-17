import 'package:flutter/material.dart';
import 'package:live_emotion_detection/home2.dart';
import 'package:live_emotion_detection/main.dart';

import 'home.dart';

void main() {
  runApp(const MyApp());
}

class Landing extends StatelessWidget {
  const Landing({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          centerTitle: true,
          title: const Text('Live Emotion Detection '),
        ),
        body: Padding(
          padding: const EdgeInsets.fromLTRB(50.0, 0.0, 50.0, 0.0),
          child: Column(
            children: [
              Image.asset(
                'assets/hehe.gif',
                width: 200,
                height: 200,
              ),
              const SizedBox(height: 35.0),
              const Text(
                'GSOC 2025 Qualification Task',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 30.0),
              Container(
                padding: EdgeInsets.all(15),
                decoration: BoxDecoration(
                  color: const Color.fromARGB(255, 191, 190, 190),
                  borderRadius: BorderRadius.circular(20.0),
                ),
                child: TextButton(
                  style: TextButton.styleFrom(
                    foregroundColor: const Color.fromARGB(255, 10, 10, 10),
                    padding: const EdgeInsets.all(16.0),
                    textStyle: const TextStyle(fontSize: 20),
                  ),
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => const Home()),
                    );
                  },
                  child: const Text('Live From Camera'),
                ),
              ),
              const SizedBox(height: 20.0),
              Container(
                padding: EdgeInsets.all(15),
                decoration: BoxDecoration(
                  color: const Color.fromARGB(255, 191, 190, 190),
                  borderRadius: BorderRadius.circular(20.0),
                ),
                child: TextButton(
                  style: TextButton.styleFrom(
                    foregroundColor: const Color.fromARGB(255, 10, 10, 10),
                    padding: const EdgeInsets.all(16.0),
                    textStyle: const TextStyle(fontSize: 20),
                  ),
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => const ImageEmotionDetector()),
                    );
                  },
                  child: const Text('Static Image'),
                ),
              ),
              const SizedBox(height: 80.0),
              const Text('Made By Md.Saif Ali Molla',
              style: TextStyle(
                fontWeight: FontWeight.w900,
                fontSize: 18,
                color: Color.fromARGB(255, 241, 18, 2)
              ),)
            ],
          ),
        ),
    );
  }
}
