# Junit on the command-line

Source: http://examples.javacodegeeks.com/core-java/junit/junit-suite-test-example/

## Installing deps

```bash
wget -O junit-4.12.jar \
http://search.maven.org/remotecontent?filepath=junit/junit/4.12/junit-4.12.jar

wget -O hamcrest-core-1.3.jar \
http://search.maven.org/remotecontent?filepath=org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar
```

## Compiling files

```bash
javac -classpath junit-4.12.jar:hamcrest-core-1.3.jar \
First.java FirstTest.java FirstTestSuite.java FirstTestSuiteRunner.java
```

## Running Test Suite

```bash
java -classpath $(pwd):junit-4.12.jar:hamcrest-core-1.3.jar FirstTestSuiteRunner
All tests finished successfully.
```
