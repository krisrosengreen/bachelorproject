Out.txt:
	echo "prevent accidental deletion" > "Out.txt"

clean:
	rm qefiles/images/* qefiles/datfiles/* qefiles/gnufiles/* qefiles/config/*
	rm -r qefiles/tmp
