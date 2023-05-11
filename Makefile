Out.txt:
	echo "prevent accidental deletion" > "Out.txt"

clean:
	rm qefiles/images/* qefiles/datfiles/* qefiles/gnufiles/*
	rm -r qefiles/tmp
