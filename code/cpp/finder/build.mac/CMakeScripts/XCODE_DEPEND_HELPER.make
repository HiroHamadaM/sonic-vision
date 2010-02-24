# DO NOT EDIT
# This makefile makes sure all linkable targets are
# up-to-date with anything they link to, avoiding a bug in XCode 1.5
all.Debug: \
	/Users/gijs/Work/sonic-vision/code/cpp/finder/build.mac/Debug/finder

all.Release: \
	/Users/gijs/Work/sonic-vision/code/cpp/finder/build.mac/Release/finder

all.MinSizeRel: \
	/Users/gijs/Work/sonic-vision/code/cpp/finder/build.mac/MinSizeRel/finder

all.RelWithDebInfo: \
	/Users/gijs/Work/sonic-vision/code/cpp/finder/build.mac/RelWithDebInfo/finder

# For each target create a dummy rule so the target does not have to exist


# Rules to remove targets that are older than anything to which they
# link.  This forces Xcode to relink the targets from scratch.  It
# does not seem to check these dependencies itself.
/Users/gijs/Work/sonic-vision/code/cpp/finder/build.mac/Debug/finder:
	/bin/rm -f /Users/gijs/Work/sonic-vision/code/cpp/finder/build.mac/Debug/finder


/Users/gijs/Work/sonic-vision/code/cpp/finder/build.mac/Release/finder:
	/bin/rm -f /Users/gijs/Work/sonic-vision/code/cpp/finder/build.mac/Release/finder


/Users/gijs/Work/sonic-vision/code/cpp/finder/build.mac/MinSizeRel/finder:
	/bin/rm -f /Users/gijs/Work/sonic-vision/code/cpp/finder/build.mac/MinSizeRel/finder


/Users/gijs/Work/sonic-vision/code/cpp/finder/build.mac/RelWithDebInfo/finder:
	/bin/rm -f /Users/gijs/Work/sonic-vision/code/cpp/finder/build.mac/RelWithDebInfo/finder


