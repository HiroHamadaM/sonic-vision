#!/usr/bin/env python

import osc
import time

def myTest():
    osc.init()
    print 'ready to send osc messages ...'
    
    for i in range(100):
        print 'sending...'
        osc.sendMsg("/left", [i], "127.0.0.1", 6666)
        osc.sendMsg("/right", [100-i], "127.0.0.1", 6666)
        #bundle = osc.createBundle()
        #osc.appendToBundle(bundle, "/test/bndlprt1", [1, 2, 3]) # 1st message appent to bundle
        #osc.appendToBundle(bundle, "/test/bndlprt2", [4, 5, 6]) # 2nd message appent to bundle
        #osc.sendBundle(bundle, "127.0.0.1", 9003) # send it to a specific ip and port
        time.sleep(0.1) # you don't need this, but otherwise we're sending as fast as possible.
    osc.dontListen() # finally close the connection bfore exiting or program

if __name__ == '__main__': myTest()














