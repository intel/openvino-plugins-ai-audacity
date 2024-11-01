/**********************************************************************

Audacity: A Digital Audio Editor

Biquad.h

Norm C
Max Maisel

***********************************************************************/

#ifndef __BIQUAD_AUDIOSR_H__
#define __BIQUAD_AUDIOSR_H__
#include <functional>
#include <limits>

#include <cstdint>
// C++ standard header <memory> with a few extensions
#include <memory>
#include <new> // align_val_t and hardware_destructive_interference_size
#include <cstdlib> // Needed for free.
#include <cmath>
#ifndef safenew
#define safenew new
#endif

namespace ovaudiosr
{
    template<typename X>
    class ArrayOf : public std::unique_ptr<X[]>
    {
    public:
        ArrayOf() {}

        template<typename Integral>
        explicit ArrayOf(Integral count, bool initialize = false)
        {
            static_assert(std::is_unsigned<Integral>::value, "Unsigned arguments only");
            reinit(count, initialize);
        }

        //ArrayOf(const ArrayOf&) = delete;
        ArrayOf(const ArrayOf&) = delete;
        ArrayOf(ArrayOf&& that)
            : std::unique_ptr < X[] >
            (std::move((std::unique_ptr < X[] >&)(that)))
        {
        }
        ArrayOf& operator= (ArrayOf&& that)
        {
            std::unique_ptr<X[]>::operator=(std::move(that));
            return *this;
        }
        ArrayOf& operator= (std::unique_ptr<X[]>&& that)
        {
            std::unique_ptr<X[]>::operator=(std::move(that));
            return *this;
        }

        template< typename Integral >
        void reinit(Integral count,
            bool initialize = false)
        {
            static_assert(std::is_unsigned<Integral>::value, "Unsigned arguments only");
            if (initialize)
                // Initialize elements (usually, to zero for a numerical type)
                std::unique_ptr<X[]>::reset(safenew X[count]{});
            else
                // Avoid the slight initialization overhead
                std::unique_ptr<X[]>::reset(safenew X[count]);
        }
    };

    /// \brief Represents a biquad digital filter.
    struct Biquad
    {
        Biquad();
        void Reset();
        void Process(const float* pfIn, float* pfOut, int iNumSamples);

        enum
        {
            /// Numerator coefficient indices
            B0 = 0, B1, B2,
            /// Denominator coefficient indices
            A1 = 0, A2,

            /// Possible filter orders for the Calc...Filter(...) functions
            MIN_Order = 1,
            MAX_Order = 10
        };

        inline float ProcessOne(float fIn)
        {
            // Biquad must use double for all calculations. Otherwise some
            // filters may have catastrophic rounding errors!
            double fOut = double(fIn) * fNumerCoeffs[B0] +
                fPrevIn * fNumerCoeffs[B1] +
                fPrevPrevIn * fNumerCoeffs[B2] -
                fPrevOut * fDenomCoeffs[A1] -
                fPrevPrevOut * fDenomCoeffs[A2];
            fPrevPrevIn = fPrevIn;
            fPrevIn = fIn;
            fPrevPrevOut = fPrevOut;
            fPrevOut = fOut;
            return fOut;
        }

        double fNumerCoeffs[3]; // B0 B1 B2
        double fDenomCoeffs[2]; // A1 A2, A0 == 1.0
        double fPrevIn;
        double fPrevPrevIn;
        double fPrevOut;
        double fPrevPrevOut;

        enum kSubTypes
        {
            kLowPass,
            kHighPass,
            nSubTypes
        };

        static ArrayOf<Biquad> CalcButterworthFilter(int order, double fn, double fc, int type);
        static ArrayOf<Biquad> CalcChebyshevType1Filter(int order, double fn, double fc, double ripple, int type);
        static ArrayOf<Biquad> CalcChebyshevType2Filter(int order, double fn, double fc, double ripple, int type);

        static void ComplexDiv(double fNumerR, double fNumerI, double fDenomR, double fDenomI,
            double* pfQuotientR, double* pfQuotientI);
        static bool BilinTransform(double fSX, double fSY, double* pfZX, double* pfZY);
        static float Calc2D_DistSqr(double fX1, double fY1, double fX2, double fY2);

        static const double s_fChebyCoeffs[MAX_Order][MAX_Order + 1];
        static double ChebyPoly(int Order, double NormFreq);
    };
}

#endif
