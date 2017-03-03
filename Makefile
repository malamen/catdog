chupalla: train.cpp

	g++ -o train.o train.cpp `pkg-config opencv --cflags --libs`
	g++ -o test_catdog.o test_catdog.cpp `pkg-config opencv --cflags --libs`