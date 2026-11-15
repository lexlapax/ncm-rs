#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static uint32_t state=0x6279ab31u;
static uint32_t random32(void) { state ^= state<<13; state ^= state>>17; state ^= state<<5; return state; }
static uint32_t raw(float x) { uint32_t u; memcpy(&u,&x,4); return u; }
static int above(float cum, float total, uint32_t bits) {
    if(cum<=0) return 0;
    uint32_t c=raw(cum), t=raw(total);
    int ce=(c>>23)&255, te=(t>>23)&255;
    uint32_t cm=(c&0x7fffff) | (ce?0x800000:0), tm=(t&0x7fffff)|0x800000;
    int shift=(ce?ce:1)-te+32;
    if(shift<0) return bits==0;
    return ((uint64_t)cm<<shift) > (uint64_t)tm*bits;
}
int main(void) {
    unsigned long long count=0,differences=0;
    for (int i=0;i<1000000;i++) {
        float total=ldexpf(1.0f+(float)(random32()&0xffffffu)*0x1p-24f,(int)(random32()%13));
        uint32_t bits=random32();
        float rounded=(float)((long double)total*bits*0x1p-32L);
        float boundaries[]={0.0f,nextafterf(rounded,-INFINITY),rounded,nextafterf(rounded,INFINITY),total};
        for(int j=0;j<5;j++) {
            float v=boundaries[j]; if(v<0||v>total) continue;
            int expected=(long double)total*bits*0x1p-32L < (long double)v;
            int actual=above(v,total,bits);
            if(expected!=actual && differences<3) printf("mismatch total=%a bits=%08x value=%a\n",total,bits,v);
            differences += (expected!=actual); count++;
        }
    }
    printf("{\"cases\":%llu,\"mismatches\":%llu}\n",count,differences);
    return differences?1:0;
}
