extern "C" {
    class A {
    public:
        A(){}
        int f(){return 9;}
    };
    void f(int x) {
        A a;
        a.f();
    }
}
